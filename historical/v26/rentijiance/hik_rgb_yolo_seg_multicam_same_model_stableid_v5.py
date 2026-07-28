
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hik_rgb_yolo_seg_singlecam_uniqueid_strict.py

单相机 RGB YOLO-seg 人体分割 + 真实场地坐标 real_xy + 严格稳定 ID + 障碍物遮挡重识别版本。

保留：
  1) 海康 MVS 实时输入；
  2) 视频文件输入，支持 OpenCV，OpenCV 失败时可用 ffmpeg 兜底；
  3) YOLO-seg person mask；
  4) mask 后处理、同人碎片合并；
  5) foot_pixel/real_xy；
  6) 单相机内部稳定 ID。

删除/不再使用：
  - tiled_infer 切片检测；
  - opencv_debug 普通摄像头；
  - Ultralytics ByteTrack/raw track id；
  - 交互式标定 UI；
  - 单人强复用、超大范围 fallback 复用。

新增：
  - local_xy 直接表示真实场地坐标 X/Y，单位米；
  - edge_related 使用当前相机图像边缘 bbox 判断，不再把全场边界误当相机边界；
  - occlusion_zones：障碍物遮挡区域配置。
  - 当旧 ID 从障碍物附近消失，并在同一障碍物另一侧短时间内重新出现时，允许在更大门限内恢复旧 ID。
  - 该规则只在配置了障碍物区域时生效，避免重新引入“远处新人继承旧 ID”的问题。

核心修复：
  原 motion_gate 版本为了避免同一个人换 ID，reappear/fallback/single_person 规则偏宽，
  导致“P1 漏检后，远处新出现的人被错误复用成 P1”。
  本版本采用 strict reappear gate：
    - 短暂漏检且位置/local_xy 接近：允许沿用旧 ID；
    - 旧 ID 不是从边界消失，并且新检测点距离旧位置太远：直接新建 ID；
    - 不再做“画面只有一个人就强制复用旧 ID”。
"""

import argparse
import csv
import ctypes
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import datetime
import queue
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

try:
    from latest_frame_shm import SharedFrameWriter
except Exception:
    # If this file is launched from the rentijiance/ subdirectory, latest_frame_shm.py
    # may live one level above the script.  Add that parent once and retry.
    try:
        _parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from latest_frame_shm import SharedFrameWriter
    except Exception:
        SharedFrameWriter = None


@dataclass
class PersonInstance:
    instance_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    mask: np.ndarray
    contours: List[np.ndarray]
    polygons: List[List[List[int]]]
    area: int
    source_ids: Optional[List[int]] = None
    foot_pixel: Optional[Tuple[int, int]] = None
    local_xy: Optional[Tuple[float, float]] = None
    local_xy_raw: Optional[Tuple[float, float]] = None
    fused_xy: Optional[Tuple[float, float]] = None
    anchor_pixel_mode: str = ""
    camera_center_weight: float = 1.0
    coord_quality: str = "unknown"
    fusion_cluster_id: Optional[int] = None
    fusion_state: str = ""
    fusion_source_cam: str = ""
    fusion_source_cams: Optional[List[str]] = None
    fusion_center_weight: float = 0.0
    fusion_outlier: bool = False
    visibility: str = "unknown"
    # local_stable_id: 当前相机内部 StrictStableIDManager 给出的单相机稳定 ID。
    # global_id_from_A: A-only anchor ReID 给出的跨相机全局 ID，只有 A 相机允许建库，B/C/D/E 只匹配。
    local_stable_id: Optional[int] = None
    global_id_from_A: Optional[int] = None
    reid_score: float = 0.0
    reid_margin: float = 0.0
    reid_source_anchor: str = ""


def now_us() -> int:
    return int(time.time() * 1_000_000)


def ensure_dir_for_file(path: str) -> None:
    if path:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)


class MvsSharedFramePublisher:
    """Publish raw MVS frames to /dev/shm for the standalone Qt/OpenGL renderer.

    This is intentionally separate from event_overlay publishing.  The MVS process
    keeps doing acquisition/sync, but the display process can read latest MVS
    frames without cv2.imshow and without CPU-side fusion.
    """

    def __init__(self, sync_cfg: Dict[str, Any], route_order: Optional[List[str]] = None):
        self.cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.enabled = _as_bool(self.cfg.get("mvs_frame_pub_enable", False), False)
        self.name_template = str(self.cfg.get("mvs_frame_pub_name_template", "mvs_latest_{camera}") or "mvs_latest_{camera}")
        self.meta_template = str(self.cfg.get("mvs_frame_pub_meta_template", "{root}/mvs_latest_{camera}.json") or "{root}/mvs_latest_{camera}.json")
        self.root = os.path.abspath(str(self.cfg.get("root", "./sync_ipc") or "./sync_ipc"))
        self.route_order = list(route_order or [])
        self.writers: Dict[str, Any] = {}
        self.writer_shapes: Dict[str, Tuple[int, int, int]] = {}
        self._warned = False
        if self.enabled and SharedFrameWriter is None:
            print("[mvs_shm][WARN] latest_frame_shm.SharedFrameWriter not available; MVS shm publishing disabled", flush=True)
            self.enabled = False
        if self.enabled:
            os.makedirs(self.root, exist_ok=True)
            print(f"[mvs_shm] enabled name_template={self.name_template!r} root={self.root}", flush=True)

    def _name_for(self, cam_id: str) -> str:
        return self.name_template.format(camera=str(cam_id), cam=str(cam_id), id=str(cam_id))

    def _meta_path_for(self, cam_id: str) -> str:
        path = self.meta_template.format(root=self.root, camera=str(cam_id), cam=str(cam_id), id=str(cam_id))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def _ensure_writer(self, cam_id: str, frame: np.ndarray) -> Optional[Any]:
        if not self.enabled or frame is None:
            return None
        if frame.ndim == 2:
            h, w = frame.shape[:2]
            c = 1
        else:
            h, w, c = frame.shape[:3]
            c = int(c)
        shape = (int(h), int(w), int(c))
        old = self.writer_shapes.get(str(cam_id))
        if str(cam_id) in self.writers and old == shape:
            return self.writers[str(cam_id)]
        try:
            if str(cam_id) in self.writers:
                try:
                    self.writers[str(cam_id)].close(unlink=True)
                except Exception:
                    pass
            name = self._name_for(str(cam_id))
            writer = SharedFrameWriter(name, width=shape[1], height=shape[0], channels=shape[2], create=True)
            self.writers[str(cam_id)] = writer
            self.writer_shapes[str(cam_id)] = shape
            print(f"[mvs_shm][{cam_id}] publishing /dev/shm/{name}.mmap shape={shape}", flush=True)
            return writer
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"[mvs_shm][{cam_id}][WARN] create writer failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    def write(self, cam_id: str, frame: np.ndarray, timestamp_us: int, fid: int, mvs_frame_num: int, mvs_meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or frame is None:
            return
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            return
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            return
        # The renderer expects 1 or 3 channels. Drop alpha if a 4-channel frame ever appears.
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = np.ascontiguousarray(arr)
        writer = self._ensure_writer(str(cam_id), arr)
        if writer is None:
            return
        try:
            writer.write(arr, timestamp_us=int(timestamp_us or now_us()), frame_id=int(fid))
            meta = dict(mvs_meta or {})
            meta.update({
                "cam_id": str(cam_id),
                "frame_id_local": int(fid),
                "shared_frame_id": int(fid),
                "mvs_frame_num": int(mvs_frame_num),
                "timestamp_us": int(timestamp_us or 0),
                "publish_wall_us": int(now_us()),
                "shape": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
                "shm_name": self._name_for(str(cam_id)),
            })
            meta_path = self._meta_path_for(str(cam_id))
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            tmp = meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, meta_path)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"[mvs_shm][{cam_id}][WARN] write failed: {type(exc).__name__}: {exc}", flush=True)

    def close(self) -> None:
        for w in list(self.writers.values()):
            try:
                w.close(unlink=False)
            except Exception:
                pass
        self.writers.clear()
        self.writer_shapes.clear()


def make_color(idx: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 0), (0, 220, 255), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 128, 0), (180, 255, 180), (255, 180, 255),
        (128, 255, 255), (255, 128, 128), (160, 160, 255), (180, 220, 80),
    ]
    return palette[idx % len(palette)]


def postprocess_mask(mask_u8: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = mask_u8.copy()
    if int(args.mask_open_iters) > 0:
        k = max(1, int(args.mask_open_kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=int(args.mask_open_iters))
    if int(args.mask_close_iters) > 0:
        k = max(1, int(args.mask_close_kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=int(args.mask_close_iters))
    if int(args.mask_erode_px) > 0:
        k = int(args.mask_erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.erode(out, kernel, iterations=1)
    if int(args.mask_dilate_px) > 0:
        k = int(args.mask_dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)
    return out


def split_person_mask_components(mask_u8: np.ndarray, args: argparse.Namespace) -> List[np.ndarray]:
    """Split one YOLO person mask into stable components when YOLO merges people.

    This is intentionally conservative.  It only runs when mask_split_components is
    enabled and returns the original mask unless the split has at least two large
    pieces.  The split is applied before stable-id assignment, so hit_judge still
    receives normal per-person masks through human_result_*.jsonl; no event/bullet
    timing code is touched.
    """
    if mask_u8 is None:
        return []
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) <= 0:
        return []
    if not bool(getattr(args, "mask_split_components", False)):
        return [mask]

    total_area = int(np.count_nonzero(mask))
    min_area_abs = int(getattr(args, "mask_split_min_area", 160))
    min_area_ratio = float(getattr(args, "mask_split_min_area_ratio", 0.08))
    min_area = max(1, max(min_area_abs, int(round(total_area * min_area_ratio))))
    erode_px = max(0, int(getattr(args, "mask_split_erode_px", 0)))
    restore_px = max(0, int(getattr(args, "mask_split_restore_dilate_px", erode_px + 1)))
    max_components = max(1, int(getattr(args, "mask_split_max_components", 4)))
    min_sum_ratio = float(getattr(args, "mask_split_min_sum_area_ratio", 0.45))

    work = mask
    if erode_px > 0:
        k = erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.erode(mask, kernel, iterations=1)

    try:
        n, labels, stats, _cent = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), 8)
    except Exception:
        return [mask]

    comps = []
    for lab in range(1, int(n)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comps.append((area, lab))
    comps.sort(reverse=True)
    comps = comps[:max_components]
    if len(comps) < 2:
        return [mask]
    if sum(a for a, _ in comps) < total_area * min_sum_ratio:
        return [mask]

    restore_kernel = None
    if restore_px > 0:
        kk = restore_px * 2 + 1
        restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))

    remaining = mask.copy()
    parts: List[np.ndarray] = []
    for _area, lab in comps:
        comp = ((labels == lab).astype(np.uint8) * 255)
        if restore_kernel is not None:
            comp = cv2.dilate(comp, restore_kernel, iterations=1)
        part = cv2.bitwise_and(mask, comp)
        part = cv2.bitwise_and(part, remaining)
        part_area = int(np.count_nonzero(part))
        if part_area < min_area:
            continue
        parts.append(part)
        remaining[part > 0] = 0

    # Keep large leftover islands, but avoid arms/noise creating fake persons.
    if int(np.count_nonzero(remaining)) >= min_area:
        try:
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats((remaining > 0).astype(np.uint8), 8)
            for lab in range(1, int(n2)):
                area = int(st2[lab, cv2.CC_STAT_AREA])
                if area >= min_area:
                    parts.append((lab2 == lab).astype(np.uint8) * 255)
        except Exception:
            pass

    if len(parts) < 2:
        return [mask]
    parts.sort(key=lambda m: int(np.count_nonzero(m)), reverse=True)
    return parts[:max_components]


def contours_to_polygons(contours: List[np.ndarray], epsilon_ratio: float) -> List[List[List[int]]]:
    polygons: List[List[List[int]]] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, float(epsilon_ratio) * float(peri))
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)
        if len(pts) >= 3:
            polygons.append([[int(x), int(y)] for x, y in pts.tolist()])
    return polygons


def bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = float(iw * ih)
    aa = float(max(1, (ax2 - ax1 + 1) * (ay2 - ay1 + 1)))
    bb = float(max(1, (bx2 - bx1 + 1) * (by2 - by1 + 1)))
    return inter / max(1.0, aa + bb - inter)


def bbox_contain_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = float(iw * ih)
    aa = float(max(1, (ax2 - ax1 + 1) * (ay2 - ay1 + 1)))
    bb = float(max(1, (bx2 - bx1 + 1) * (by2 - by1 + 1)))
    return inter / max(1.0, min(aa, bb))


def bbox_gap_overlap_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1) + 1)
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1) + 1)
    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2) - 1)
    gap_y = max(0, max(ay1, by1) - min(ay2, by2) - 1)
    return gap_x, gap_y, overlap_x, overlap_y


def bbox_center_xyxy(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def size_ratio(a: int, b: int) -> float:
    return float(max(a, b)) / float(max(1, min(a, b)))


def build_person_from_union_mask(mask: np.ndarray, members: List[PersonInstance], args: argparse.Namespace) -> Optional[PersonInstance]:
    area = int(np.count_nonzero(mask))
    if area < int(args.min_mask_area):
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= float(args.min_contour_area)]
    if not contours:
        return None
    polygons = contours_to_polygons(contours, args.poly_epsilon_ratio)
    if not polygons:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    confidence = max(float(m.confidence) for m in members)
    source_ids: List[int] = []
    for m in members:
        source_ids.extend([int(v) for v in (m.source_ids or [m.instance_id])])
    source_ids = sorted(set(source_ids))
    return PersonInstance(
        instance_id=int(source_ids[0]) if source_ids else int(members[0].instance_id),
        confidence=confidence,
        bbox_xyxy=bbox,
        mask=mask,
        contours=contours,
        polygons=polygons,
        area=area,
        source_ids=source_ids,
    )


def should_merge_same_person(a: PersonInstance, b: PersonInstance, args: argparse.Namespace) -> bool:
    """智能碎片合并：保留原版本“合并同一个人的碎片”的稳定性，
    但禁止把两个完整人体粗暴合并成一个 ID。

    旧版 merge_same_person 对 ID 稳定有帮助，因为 YOLO-seg 有时会把同一个人拆成头/手/身体碎片。
    但旧版也会在两个人靠近时把两个完整人体合成一个人。
    这里改成：只有“明显小碎片 + 靠近/包含/接触大人体”才合并；两个面积相近、都像完整人的目标不合并。
    """
    area_a, area_b = max(1, int(a.area)), max(1, int(b.area))
    small_ratio = min(area_a, area_b) / max(area_a, area_b)
    iou = bbox_iou_xyxy(a.bbox_xyxy, b.bbox_xyxy)
    contain = bbox_contain_xyxy(a.bbox_xyxy, b.bbox_xyxy)

    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    aw, ah = max(1, ax2 - ax1 + 1), max(1, ay2 - ay1 + 1)
    bw, bh = max(1, bx2 - bx1 + 1), max(1, by2 - by1 + 1)
    height_ratio = float(min(ah, bh)) / float(max(1, max(ah, bh)))
    gap_x, gap_y, overlap_x, overlap_y = bbox_gap_overlap_xyxy(a.bbox_xyxy, b.bbox_xyxy)
    y_overlap_ratio = overlap_y / max(1.0, float(min(ah, bh)))
    x_overlap_ratio = overlap_x / max(1.0, float(min(aw, bw)))
    ca, cb = bbox_center_xyxy(a.bbox_xyxy), bbox_center_xyxy(b.bbox_xyxy)
    center_dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))

    # 可配置参数。默认值偏保守：宁愿少合并，也不要把两个人合成一个 ID。
    smart_enable = bool(getattr(args, "merge_smart_fragment_enable", True))
    if not smart_enable:
        if iou >= float(args.merge_iou_thresh):
            return True
        if contain >= float(args.merge_contain_thresh):
            return True
        if small_ratio > float(args.merge_small_area_ratio):
            return False

    fragment_ratio = float(getattr(args, "merge_fragment_area_ratio", 0.58))
    full_pair_ratio = float(getattr(args, "merge_full_pair_area_ratio", 0.62))
    full_pair_height_ratio = float(getattr(args, "merge_full_pair_height_ratio", 0.68))
    full_pair_min_area_ratio = float(getattr(args, "merge_full_pair_min_area_ratio", 0.35))
    strong_iou = float(getattr(args, "merge_strong_iou_thresh", 0.18))
    strong_contain = float(getattr(args, "merge_strong_contain_thresh", 0.62))
    max_fragment_center = float(getattr(args, "merge_fragment_center_dist_px", getattr(args, "merge_center_dist_px", 95.0)))

    # 两个目标面积/高度都比较接近时，更像“两个人靠近”，而不是“同一个人的小碎片”。
    # 这里要更保守：除非出现非常强的包含/重叠，否则不合并，避免开头检测框/轮廓被合成一大块。
    partial_min_h = float(getattr(args, "partial_body_min_h_px", 80.0))
    both_look_like_person = (min(ah, bh) >= partial_min_h and
                             height_ratio >= full_pair_height_ratio and
                             small_ratio >= full_pair_min_area_ratio)
    # v12:多人站得很近时，宁愿保留两个候选，也不要把两个完整人体合成一个大 mask。
    # 只有非常强的重叠/包含才允许合并；普通接触、轻微重叠不再合并。
    if smart_enable and (small_ratio >= full_pair_ratio or both_look_like_person):
        return bool(iou >= max(0.32, strong_iou * 1.45) or contain >= min(0.94, strong_contain + 0.14))

    # 小碎片被大框明显包含，或者和大框高度/宽度有重叠，才允许合并。
    if contain >= float(args.merge_contain_thresh) and small_ratio <= float(args.merge_small_area_ratio):
        return True
    if iou >= float(args.merge_iou_thresh) and small_ratio <= float(args.merge_small_area_ratio):
        return True

    # 膨胀后接触只作为“碎片合并”条件，不作为“完整人体合并”条件。
    dilate_px = max(0, int(args.merge_mask_dilate_px))
    touched = False
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ma = cv2.dilate(a.mask, kernel, iterations=1)
        mb = cv2.dilate(b.mask, kernel, iterations=1)
        touched = int(np.count_nonzero((ma > 0) & (mb > 0))) > 0
    else:
        touched = int(np.count_nonzero((a.mask > 0) & (b.mask > 0))) > 0

    if small_ratio <= fragment_ratio:
        if touched and gap_x <= int(args.merge_gap_px) and gap_y <= int(args.merge_gap_px):
            return True
        if center_dist <= max_fragment_center and (x_overlap_ratio >= float(args.merge_min_x_overlap_ratio) or y_overlap_ratio >= float(args.merge_min_y_overlap_ratio)):
            return True

    return False



def merge_same_person_instances(persons: List[PersonInstance], args: argparse.Namespace) -> List[PersonInstance]:
    if not persons or len(persons) <= 1 or not bool(getattr(args, "merge_same_person", False)):
        return persons
    n = len(persons)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_same_person(persons[i], persons[j], args):
                union(i, j)
    groups: Dict[int, List[PersonInstance]] = {}
    for i, p in enumerate(persons):
        groups.setdefault(find(i), []).append(p)
    merged: List[PersonInstance] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        union_mask = np.zeros_like(members[0].mask, dtype=np.uint8)
        for m in members:
            union_mask = cv2.bitwise_or(union_mask, m.mask)
        if int(args.merge_close_kernel) > 1 and int(args.merge_close_iters) > 0:
            k = int(args.merge_close_kernel)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            union_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel, iterations=int(args.merge_close_iters))
        p = build_person_from_union_mask(union_mask, members, args)
        if p is not None:
            merged.append(p)
    merged.sort(key=lambda p: p.area, reverse=True)
    return merged


def compute_color_hist(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> Optional[np.ndarray]:
    try:
        if frame_bgr is None or mask_u8 is None or int(np.count_nonzero(mask_u8)) < 30:
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = (mask_u8 > 0).astype(np.uint8) * 255
        hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None).flatten().astype(np.float32)
        return hist
    except Exception:
        return None


def hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    try:
        sim = float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))
    except Exception:
        return 0.0


def _l2_normalize_feature(v: np.ndarray) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size <= 0 or not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return None
    return (v / n).astype(np.float32)



def _masked_hs_hist_from_hsv(hsv_image: np.ndarray, mask_u8: np.ndarray, h_bins: int = 18, s_bins: int = 8) -> Optional[np.ndarray]:
    """A-anchor ReID HSV 特征：输入已经转换好的 HSV 图，避免每个人重复 BGR->HSV。"""
    try:
        if hsv_image is None or mask_u8 is None or int(np.count_nonzero(mask_u8)) < 30:
            return None
        m = (mask_u8 > 0).astype(np.uint8) * 255
        hist = cv2.calcHist([hsv_image], [0, 1], m, [int(h_bins), int(s_bins)], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
        hist = np.sqrt(np.maximum(hist, 0.0))  # Root-Histogram，减弱大面积颜色支配
        return _l2_normalize_feature(hist)
    except Exception:
        return None


def _masked_hs_hist(frame_bgr: np.ndarray, mask_u8: np.ndarray, h_bins: int = 18, s_bins: int = 8) -> Optional[np.ndarray]:
    """A-anchor ReID 用的轻量外观特征：HSV 的 H/S 直方图。"""
    try:
        if frame_bgr is None or mask_u8 is None or int(np.count_nonzero(mask_u8)) < 30:
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return _masked_hs_hist_from_hsv(hsv, mask_u8, h_bins=h_bins, s_bins=s_bins)
    except Exception:
        return None


def extract_a_anchor_feature(frame_bgr: np.ndarray,
                             person: PersonInstance,
                             args: argparse.Namespace,
                             hsv_frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """从人体 mask 内提取 A-only anchor ReID 特征。

    这个版本仍使用 HSV whole/upper/lower 三段颜色特征，保持原有 ID 判定逻辑；
    但支持两项加速：
      1) 调用方可传入每帧只转换一次的 hsv_frame，避免每个人重复 cvtColor；
      2) 默认只在人体 bbox ROI 内计算直方图，避免整张原图做 mask histogram。
    后续如果要换成 OSNet/FastReID，只需要替换这个函数返回的 feature。
    """
    if frame_bgr is None or person is None or person.mask is None:
        return None
    mask = (person.mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < int(getattr(args, "a_anchor_reid_min_feature_area", 80)):
        return None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person.bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    try:
        hsv = hsv_frame if hsv_frame is not None else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return None

    use_roi = bool(getattr(args, "a_anchor_reid_feature_roi_enable", True))
    if x2 <= x1 or y2 <= y1:
        return _masked_hs_hist_from_hsv(hsv, mask)

    body_h = max(1, y2 - y1 + 1)
    upper_y2 = min(y2, y1 + int(round(body_h * float(getattr(args, "a_anchor_reid_upper_ratio", 0.58)))))
    lower_y1 = max(y1, y1 + int(round(body_h * float(getattr(args, "a_anchor_reid_lower_start_ratio", 0.42)))))

    if use_roi:
        # 只对 bbox ROI 做直方图，和整图 mask 结果等价近似，但 CPU 开销小很多。
        whole = _masked_hs_hist_from_hsv(hsv[y1:y2 + 1, x1:x2 + 1], mask[y1:y2 + 1, x1:x2 + 1])
        upper = _masked_hs_hist_from_hsv(hsv[y1:upper_y2 + 1, x1:x2 + 1], mask[y1:upper_y2 + 1, x1:x2 + 1])
        lower = _masked_hs_hist_from_hsv(hsv[lower_y1:y2 + 1, x1:x2 + 1], mask[lower_y1:y2 + 1, x1:x2 + 1])
    else:
        whole = _masked_hs_hist_from_hsv(hsv, mask)
        upper_mask = np.zeros_like(mask, dtype=np.uint8)
        upper_mask[y1:upper_y2 + 1, x1:x2 + 1] = mask[y1:upper_y2 + 1, x1:x2 + 1]
        lower_mask = np.zeros_like(mask, dtype=np.uint8)
        lower_mask[lower_y1:y2 + 1, x1:x2 + 1] = mask[lower_y1:y2 + 1, x1:x2 + 1]
        upper = _masked_hs_hist_from_hsv(hsv, upper_mask)
        lower = _masked_hs_hist_from_hsv(hsv, lower_mask)

    parts = []
    for feat, weight in [
        (whole, float(getattr(args, "a_anchor_reid_whole_weight", 0.45))),
        (upper, float(getattr(args, "a_anchor_reid_upper_weight", 0.35))),
        (lower, float(getattr(args, "a_anchor_reid_lower_weight", 0.20))),
    ]:
        if feat is None:
            parts.append(np.zeros(18 * 8, dtype=np.float32))
        else:
            parts.append(feat.astype(np.float32) * max(0.0, weight))
    return _l2_normalize_feature(np.concatenate(parts, axis=0))


# =============================================================================
# Optional open-source deep ReID backend: Torchreid OSNet
# =============================================================================

_A_ANCHOR_DEEP_REID_EXTRACTOR: Optional[Any] = None
_A_ANCHOR_DEEP_REID_WARNED = False


def _a_anchor_reid_backend(args: argparse.Namespace) -> str:
    return str(getattr(args, "a_anchor_reid_backend", "hsv") or "hsv").strip().lower()


def _a_anchor_deep_backend_enabled(args: argparse.Namespace) -> bool:
    return _a_anchor_reid_backend(args) in {"osnet", "torchreid", "deep", "deep_osnet"}


def _crop_person_for_deep_reid(frame_bgr: np.ndarray, person: PersonInstance, args: argparse.Namespace) -> Optional[np.ndarray]:
    """按 bbox 裁剪人体。可选用 mask 把背景置为中性灰，减少旁边玩家干扰。"""
    if frame_bgr is None or person is None:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person.bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    expand = float(getattr(args, "a_anchor_reid_osnet_crop_expand", 0.08))
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    ex = int(round(bw * expand))
    ey = int(round(bh * expand))
    x1 = max(0, x1 - ex)
    y1 = max(0, y1 - ey)
    x2 = min(w - 1, x2 + ex)
    y2 = min(h - 1, y2 + ey)
    crop = frame_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    if crop.size <= 0:
        return None
    if bool(getattr(args, "a_anchor_reid_osnet_mask_crop", True)) and person.mask is not None:
        try:
            mask_roi = (person.mask[y1:y2 + 1, x1:x2 + 1] > 0)
            if mask_roi.shape[:2] == crop.shape[:2] and int(np.count_nonzero(mask_roi)) >= 20:
                bg = int(getattr(args, "a_anchor_reid_osnet_mask_bg", 114))
                crop[~mask_roi] = (bg, bg, bg)
        except Exception:
            pass
    return crop


class TorchreidOSNetExtractor:
    """把 Torchreid 的 OSNet 当作 A-anchor 的 deep feature extractor。

    本地权重版的关键点：
      1) 支持 osnet_source_dir，程序可自动把 deep-person-reid 源码目录加入 sys.path，
         不再要求每次手动 export PYTHONPATH；
      2) local_only=True 时必须提供 osnet_weights 本地文件；
      3) build_model 永远使用 pretrained=False，禁止 torchreid 启动时自动联网下载权重；
      4) 权重不存在或加载失败时直接报清楚错误，不再假死卡在下载阶段。
    """
    def __init__(self, args: argparse.Namespace):
        self.model_name = str(getattr(args, "a_anchor_reid_osnet_model", "osnet_x0_25") or "osnet_x0_25")
        self.input_h = int(getattr(args, "a_anchor_reid_osnet_input_h", 256))
        self.input_w = int(getattr(args, "a_anchor_reid_osnet_input_w", 128))
        self.batch_size = max(1, int(getattr(args, "a_anchor_reid_osnet_batch_size", 16)))
        self.use_half = bool(getattr(args, "a_anchor_reid_osnet_half", False))
        self.device_name = str(getattr(args, "a_anchor_reid_osnet_device", "auto") or "auto")
        self.source_dir = str(getattr(args, "a_anchor_reid_osnet_source_dir", "") or "").strip()
        self.weights = str(getattr(args, "a_anchor_reid_osnet_weights", "") or "").strip()
        self.local_only = bool(getattr(args, "a_anchor_reid_osnet_local_only", True))

        if self.source_dir:
            src_dir = os.path.abspath(os.path.expanduser(self.source_dir))
            if not os.path.isdir(src_dir):
                raise FileNotFoundError(f"torchreid source_dir not found: {src_dir}")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

        if self.weights:
            self.weights = os.path.abspath(os.path.expanduser(self.weights))
        if self.local_only:
            if not self.weights:
                raise FileNotFoundError(
                    "OSNet local_only=true but osnet_weights is empty. "
                    "请先下载 OSNet 权重到本地，并在 JSON 的 common.a_anchor_reid.osnet_weights 中填写绝对路径。"
                )
            if not os.path.isfile(self.weights):
                raise FileNotFoundError(
                    f"OSNet local weights not found: {self.weights}. "
                    "本地权重版不会联网下载；请检查权重文件路径。"
                )

        import torch  # lazy import，避免 HSV 模式下强依赖 torch/torchreid
        import torchreid
        self.torch = torch
        self.torchreid = torchreid
        if self.device_name.lower() in {"", "auto"}:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)

        # 本地权重版禁止 pretrained=True，避免 torchreid 在启动时自动联网下载权重导致程序卡死。
        pretrained_flag = False if self.local_only or self.weights else bool(getattr(args, "a_anchor_reid_osnet_pretrained", False))
        try:
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1000,
                loss="softmax",
                pretrained=bool(pretrained_flag),
                use_gpu=(self.device.type == "cuda"),
            )
        except TypeError:
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1000,
                pretrained=bool(pretrained_flag),
            )

        loaded_keys = 0
        if self.weights:
            loaded_keys = self._load_local_weights(self.weights)
            if loaded_keys <= 0:
                raise RuntimeError(f"OSNet local weights loaded 0 parameter keys: {self.weights}")

        self.model.to(self.device)
        self.model.eval()
        if self.use_half and self.device.type == "cuda":
            self.model.half()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        print(
            f"[A-anchor][OSNet] loaded LOCAL backend=torchreid model={self.model_name} "
            f"device={self.device} input={self.input_h}x{self.input_w} "
            f"source_dir={self.source_dir or '(python path)'} weights={self.weights or '(none)'} "
            f"loaded_keys={loaded_keys} pretrained_download=disabled",
            flush=True,
        )

    def _load_local_weights(self, path: str) -> int:
        """只从本地文件加载权重，跳过 classifier 尺寸不匹配参数。"""
        torch = self.torch
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported OSNet checkpoint format: {path}")

        model_state = self.model.state_dict()
        filtered = {}
        skipped = 0
        for k, v in state.items():
            kk = str(k)
            for prefix in ["module.", "model."]:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            if kk in model_state and hasattr(v, "shape") and tuple(model_state[kk].shape) == tuple(v.shape):
                filtered[kk] = v
            else:
                skipped += 1
        self.model.load_state_dict(filtered, strict=False)
        print(
            f"[A-anchor][OSNet] local weights loaded: path={path} matched_keys={len(filtered)} skipped_keys={skipped}",
            flush=True,
        )
        return int(len(filtered))

    def _prep_one(self, crop_bgr: np.ndarray) -> Optional[Any]:
        try:
            if crop_bgr is None or crop_bgr.size <= 0:
                return None
            crop = cv2.resize(crop_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            arr = np.ascontiguousarray(crop.transpose(2, 0, 1), dtype=np.float32) / 255.0
            return self.torch.from_numpy(arr)
        except Exception:
            return None

    def extract_batch(self, frame_bgr: np.ndarray, persons: List[PersonInstance], args: argparse.Namespace) -> List[Optional[np.ndarray]]:
        crops: List[Any] = []
        out_index: List[int] = []
        results: List[Optional[np.ndarray]] = [None for _ in persons]
        for i, p in enumerate(persons):
            crop = _crop_person_for_deep_reid(frame_bgr, p, args)
            ten = self._prep_one(crop) if crop is not None else None
            if ten is not None:
                crops.append(ten)
                out_index.append(i)
        if not crops:
            return results
        torch = self.torch
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                part = crops[start:start + self.batch_size]
                idxs = out_index[start:start + self.batch_size]
                batch = torch.stack(part, dim=0).to(self.device, non_blocking=True)
                batch = (batch - self.mean) / self.std
                if self.use_half and self.device.type == "cuda":
                    batch = batch.half()
                feats = self.model(batch)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
                feats_np = feats.detach().cpu().numpy().astype(np.float32)
                for j, idx in enumerate(idxs):
                    results[idx] = _l2_normalize_feature(feats_np[j])
        return results

    def extract_one(self, frame_bgr: np.ndarray, person: PersonInstance, args: argparse.Namespace) -> Optional[np.ndarray]:
        return self.extract_batch(frame_bgr, [person], args)[0]


def _get_deep_reid_extractor(args: argparse.Namespace) -> Optional[TorchreidOSNetExtractor]:
    global _A_ANCHOR_DEEP_REID_EXTRACTOR, _A_ANCHOR_DEEP_REID_WARNED
    if _A_ANCHOR_DEEP_REID_EXTRACTOR is not None:
        return _A_ANCHOR_DEEP_REID_EXTRACTOR
    try:
        _A_ANCHOR_DEEP_REID_EXTRACTOR = TorchreidOSNetExtractor(args)
        return _A_ANCHOR_DEEP_REID_EXTRACTOR
    except Exception as e:
        msg = ("[A-anchor][OSNet][WARN] failed to initialize torchreid OSNet: "
               f"{type(e).__name__}: {e}. ")
        required = bool(getattr(args, "a_anchor_reid_osnet_required", False))
        fallback = bool(getattr(args, "a_anchor_reid_osnet_fallback_to_hsv", True))
        if required and not fallback:
            raise RuntimeError(msg + "Install torchreid/torch, set osnet_source_dir, and set a valid local osnet_weights; or set backend='hsv'.")
        if not _A_ANCHOR_DEEP_REID_WARNED:
            print(msg + "Fallback to HSV ReID for now. 本地权重版需要 osnet_source_dir 和 osnet_weights 都正确；成功时会出现 '[A-anchor][OSNet] loaded LOCAL'.", flush=True)
            _A_ANCHOR_DEEP_REID_WARNED = True
        return None


def extract_a_anchor_deep_feature(frame_bgr: np.ndarray, person: PersonInstance, args: argparse.Namespace) -> Optional[np.ndarray]:
    ext = _get_deep_reid_extractor(args)
    if ext is None:
        return None
    try:
        return ext.extract_one(frame_bgr, person, args)
    except Exception as e:
        global _A_ANCHOR_DEEP_REID_WARNED
        if not _A_ANCHOR_DEEP_REID_WARNED:
            print(f"[A-anchor][OSNet][WARN] feature extraction failed: {type(e).__name__}: {e}; fallback to HSV if enabled.", flush=True)
            _A_ANCHOR_DEEP_REID_WARNED = True
        return None

def feature_cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    try:
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        if aa.shape != bb.shape:
            return 0.0
        sim = float(np.dot(aa, bb) / max(1e-8, float(np.linalg.norm(aa) * np.linalg.norm(bb))))
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


def anchor_person_shape_stats(p: PersonInstance, frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
    """A-anchor 建库质量判断用的人体形状统计。"""
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in p.bbox_xyxy]
    bw = max(1, int(x2 - x1 + 1))
    bh = max(1, int(y2 - y1 + 1))
    frame_area = max(1, int(fw) * int(fh))
    bbox_area = max(1, bw * bh)
    mask_area = max(1, int(getattr(p, "area", 0) or 0))
    return {
        "bbox_w": float(bw),
        "bbox_h": float(bh),
        "bbox_aspect": float(bw) / float(max(1, bh)),
        "bbox_width_ratio": float(bw) / float(max(1, fw)),
        "bbox_height_ratio": float(bh) / float(max(1, fh)),
        "mask_area_ratio": float(mask_area) / float(frame_area),
        "mask_fill_ratio": float(mask_area) / float(bbox_area),
    }


def is_anchor_group_candidate(p: PersonInstance, frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> bool:
    """判断 A 相机当前的一个 detection 是否很可能是多人粘连成的一个 mask。

    解决用户反馈的开局场景：三个人站在一起时 YOLO-seg 可能只输出一个大 mask，
    如果这时立即把它注册成 P1，后面三个人分开后，三个人都会继承/匹配这个 P1。

    这里不会影响 YOLO 检测，只影响 A-anchor ReID 是否允许把这个大 mask 建入身份库。
    """
    if not bool(getattr(args, "a_anchor_reid_group_guard_enable", True)):
        return False
    if p is None:
        return False
    st = anchor_person_shape_stats(p, frame_shape)
    aspect = float(st["bbox_aspect"])
    width_ratio = float(st["bbox_width_ratio"])
    area_ratio = float(st["mask_area_ratio"])
    min_w_ratio = float(getattr(args, "a_anchor_reid_group_guard_min_width_ratio", 0.10))
    max_aspect = float(getattr(args, "a_anchor_reid_group_guard_single_max_aspect", 0.78))
    max_width_ratio = float(getattr(args, "a_anchor_reid_group_guard_single_max_width_ratio", 0.18))
    max_area_ratio = float(getattr(args, "a_anchor_reid_group_guard_single_max_area_ratio", 0.018))

    # 正常单人一般是高瘦形状；多人横向粘连时 bbox 会明显变宽。
    wide_like_group = (width_ratio >= min_w_ratio and aspect >= max_aspect)
    too_wide = width_ratio >= max_width_ratio
    too_large = area_ratio >= max_area_ratio
    return bool(wide_like_group or too_wide or too_large)


class AAnchorReIDManager:
    """A 相机作为唯一身份库的跨相机 ID 测试管理器。

    规则保持不变：
      1) 只有 anchor_camera，默认 A，相机允许注册 / 更新身份库；
      2) B/C/D/E 只能匹配 A 身份库，匹配不到就保持 unknown；
      3) 不覆盖单相机 StrictStableIDManager 的 local_stable_id，只额外写 global_id_from_A；
      4) 低频 ReID：YOLO mask 每帧更新，但 A-anchor 外观特征按需/低频计算，已确认 local_id 中间帧只保持 P 标签。
    """
    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(getattr(args, "a_anchor_reid_enable", False))
        self.anchor_camera = str(getattr(args, "a_anchor_reid_anchor_camera", "A") or "A")
        self.allow_non_anchor_enroll = bool(getattr(args, "a_anchor_reid_allow_non_anchor_enroll", False))
        self.next_global_id = int(getattr(args, "a_anchor_reid_start_id", 1))
        self.max_ids = int(getattr(args, "a_anchor_reid_max_ids", 64))
        self.bank: Dict[int, Dict[str, Any]] = {}
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        # 每个相机 local_stable_id 的 ReID 缓存。只缓存标签/分数，不缓存 mask/bbox，避免旧框画到新帧。
        self.assignment_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # A 相机开局多人粘连保护：记录最近一次 anchor 只看到一个大目标的情况，
        # 后面如果它分裂成多个 detection，就重置/重新分配唯一 P1/P2/P3。
        self.last_anchor_single: Optional[Dict[str, Any]] = None

    def bank_size(self) -> int:
        return int(len(self.bank))

    def _new_global_id(self) -> int:
        gid = int(self.next_global_id)
        self.next_global_id += 1
        return gid

    def _local_id_of(self, p: PersonInstance) -> int:
        return int(getattr(p, "local_stable_id", None) or p.instance_id)

    def _cache_key(self, cam_id: str, local_id: int) -> Tuple[str, int]:
        return (str(cam_id), int(local_id))

    def _lowfreq_enabled(self, args: argparse.Namespace) -> bool:
        return bool(getattr(args, "a_anchor_reid_lowfreq_enable", True))

    def _cache_alive(self, rec: Optional[Dict[str, Any]], ts_us: int, args: argparse.Namespace) -> bool:
        if not isinstance(rec, dict):
            return False
        max_ms = float(getattr(args, "a_anchor_reid_cache_hold_max_ms", 30000.0))
        if max_ms <= 0:
            return True
        last_us = int(rec.get("last_seen_us", rec.get("last_reid_us", ts_us)) or ts_us)
        return ((int(ts_us) - last_us) / 1000.0) <= max_ms

    def _remember_assignment(self,
                             key: Tuple[str, int],
                             gid: Optional[int],
                             score: float,
                             margin: float,
                             frame_id: int,
                             ts_us: int,
                             reid_checked: bool = False,
                             bank_updated: bool = False,
                             status: str = "confirmed") -> None:
        rec = self.assignment_cache.get(key, {})
        if gid is not None and int(gid) > 0:
            rec["gid"] = int(gid)
            self.local_to_global[key] = int(gid)
        rec.pop("pending_gid", None)
        rec.pop("pending_hits", None)
        rec.pop("pending_score", None)
        rec.pop("pending_margin", None)
        rec["score"] = float(score)
        rec["margin"] = float(margin)
        rec["last_seen_frame"] = int(frame_id)
        rec["last_seen_us"] = int(ts_us)
        rec["status"] = str(status)
        if reid_checked:
            rec["last_reid_frame"] = int(frame_id)
            rec["last_reid_us"] = int(ts_us)
        if bank_updated:
            rec["last_update_frame"] = int(frame_id)
            rec["last_update_us"] = int(ts_us)
        self.assignment_cache[key] = rec

    def _remember_unknown(self, key: Tuple[str, int], frame_id: int, ts_us: int, score: float = 0.0, margin: float = 0.0) -> None:
        rec = self.assignment_cache.get(key, {})
        rec.pop("gid", None)
        rec.pop("pending_gid", None)
        rec.pop("pending_hits", None)
        rec.pop("pending_score", None)
        rec.pop("pending_margin", None)
        rec["score"] = float(score)
        rec["margin"] = float(margin)
        rec["last_seen_frame"] = int(frame_id)
        rec["last_seen_us"] = int(ts_us)
        rec["last_reid_frame"] = int(frame_id)
        rec["last_reid_us"] = int(ts_us)
        rec["status"] = "unknown"
        self.assignment_cache[key] = rec
        self.local_to_global.pop(key, None)

    def _clear_assignment(self, key: Tuple[str, int]) -> None:
        self.local_to_global.pop(key, None)
        self.assignment_cache.pop(key, None)

    def _record_bank_anchor_state(self, gid: Optional[int], p: PersonInstance, frame_id: int, ts_us: int) -> None:
        """记录 A 相机当前帧中每个 P 的场地坐标。

        这是给 B/C/D/E 查询时使用的“以 A 相机为准”的位置约束。
        只记录坐标和时间，不记录 mask/bbox，所以不会把旧人体范围画到新帧上。
        """
        try:
            if gid is None or int(gid) not in self.bank or p is None or p.local_xy is None:
                return
            rec = self.bank[int(gid)]
            rec["last_local_xy"] = (float(p.local_xy[0]), float(p.local_xy[1]))
            rec["last_anchor_frame"] = int(frame_id)
            rec["last_anchor_ts_us"] = int(ts_us)
        except Exception:
            return

    def _local_xy_gate_score(self,
                             p: Optional[PersonInstance],
                             rec: Dict[str, Any],
                             frame_id: int,
                             args: argparse.Namespace) -> Tuple[bool, Optional[float], Optional[float]]:
        """用 A 相机当前 P 的 local_xy 约束查询相机的候选 P。

        如果 A 和查询相机都能投影到同一场地坐标系，那么同一个玩家在同一帧的 local_xy
        应该接近。这个约束可以避免开局 E 相机把 P1/P3 错配成 A 库里的 P2。
        缺坐标或 A 坐标太旧时不强制过滤，避免因为某一路标定/遮挡导致全变 UNK。
        """
        if not bool(getattr(args, "a_anchor_reid_use_anchor_local_xy_gate", True)):
            return True, None, None
        if p is None or p.local_xy is None or not isinstance(rec, dict):
            return True, None, None
        anchor_xy = rec.get("last_local_xy", None)
        if anchor_xy is None:
            return True, None, None
        try:
            ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
            qx, qy = float(p.local_xy[0]), float(p.local_xy[1])
        except Exception:
            return True, None, None
        max_age = int(getattr(args, "a_anchor_reid_local_xy_max_age_frames", 3))
        last_f = rec.get("last_anchor_frame", None)
        if last_f is not None and max_age >= 0:
            try:
                if int(frame_id) - int(last_f) > max_age:
                    return True, None, None
            except Exception:
                pass
        dist = float(np.hypot(qx - ax, qy - ay))
        hard = float(getattr(args, "a_anchor_reid_local_xy_hard_dist", 6.0))
        soft = float(getattr(args, "a_anchor_reid_local_xy_soft_dist", 2.5))
        if hard > 0 and dist > hard:
            return False, 0.0, dist
        if soft <= 1e-6:
            return True, 1.0, dist
        pos_score = max(0.0, min(1.0, 1.0 - dist / max(1e-6, soft)))
        return True, float(pos_score), dist

    def _remember_pending_match(self,
                                key: Tuple[str, int],
                                gid: int,
                                score: float,
                                margin: float,
                                frame_id: int,
                                ts_us: int,
                                args: argparse.Namespace) -> bool:
        """新 local_id 首次匹配时先做临时确认，连续命中同一 P 后再正式显示。

        这样 E/B/C/D 相机开局不会因为一次模糊 OSNet 匹配就立刻显示错 P。
        返回 True 表示达到确认次数，可以正式写入 local_id -> P 映射。
        """
        need = max(1, int(getattr(args, "a_anchor_reid_query_confirm_hits", 1)))
        if need <= 1:
            return True
        rec = self.assignment_cache.get(key, {})
        prev_gid = rec.get("pending_gid", None) if isinstance(rec, dict) else None
        try:
            hits = int(rec.get("pending_hits", 0)) if int(prev_gid) == int(gid) else 0
        except Exception:
            hits = 0
        hits += 1
        rec["pending_gid"] = int(gid)
        rec["pending_hits"] = int(hits)
        rec["pending_score"] = float(score)
        rec["pending_margin"] = float(margin)
        rec["last_seen_frame"] = int(frame_id)
        rec["last_seen_us"] = int(ts_us)
        rec["last_reid_frame"] = int(frame_id)
        rec["last_reid_us"] = int(ts_us)
        rec["status"] = "pending_confirm"
        self.assignment_cache[key] = rec
        return hits >= need

    def _extract_feature_cached(self,
                                frame: np.ndarray,
                                p: PersonInstance,
                                args: argparse.Namespace,
                                hsv_holder: Dict[str, Any]) -> Optional[np.ndarray]:
        # backend=osnet/torchreid 时使用开源 OSNet 深度特征；否则保持原 HSV 颜色直方图。
        # 注意：只替换 A-anchor 的外观特征，不改变 A 建库、B/C/D/E 查询、低频缓存和同帧唯一分配逻辑。
        if _a_anchor_deep_backend_enabled(args):
            feat = extract_a_anchor_deep_feature(frame, p, args)
            if feat is not None:
                return feat
            if not bool(getattr(args, "a_anchor_reid_osnet_fallback_to_hsv", True)):
                return None
        try:
            if hsv_holder.get("hsv") is None:
                hsv_holder["hsv"] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return extract_a_anchor_feature(frame, p, args, hsv_frame=hsv_holder.get("hsv"))
        except Exception:
            return extract_a_anchor_feature(frame, p, args)

    def _match_feature(self,
                       feat: Optional[np.ndarray],
                       exclude_gids: Optional[set] = None,
                       allow_gids: Optional[set] = None,
                       query_person: Optional[PersonInstance] = None,
                       query_frame_id: int = 0,
                       args: Optional[argparse.Namespace] = None,
                       apply_local_xy_gate: bool = False) -> Tuple[Optional[int], float, float]:
        if feat is None or not self.bank:
            return None, 0.0, 0.0
        exclude_gids = set(exclude_gids or set())
        allow_gids = set(allow_gids) if allow_gids is not None else None
        scores: List[Tuple[float, int]] = []
        for gid, rec in self.bank.items():
            gid_i = int(gid)
            if gid_i in exclude_gids:
                continue
            if allow_gids is not None and gid_i not in allow_gids:
                continue
            raw_sim = feature_cosine_similarity(feat, rec.get("feature"))
            final_score = float(raw_sim)
            if apply_local_xy_gate and args is not None:
                ok, pos_score, _dist = self._local_xy_gate_score(query_person, rec, int(query_frame_id), args)
                if not ok:
                    continue
                if pos_score is not None:
                    w = float(getattr(args, "a_anchor_reid_local_xy_score_weight", 0.22))
                    w = max(0.0, min(0.85, w))
                    final_score = (1.0 - w) * float(raw_sim) + w * float(pos_score)
            scores.append((float(final_score), gid_i))
        if not scores:
            return None, 0.0, 0.0
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_gid = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        return int(best_gid), float(best_score), float(best_score - second)

    def _update_bank(self, gid: int, feat: Optional[np.ndarray], cam_id: str, ts_us: int, local_id: int, args: argparse.Namespace) -> None:
        if feat is None:
            return
        if gid in self.bank and bool(getattr(args, "a_anchor_reid_anchor_update_self_check", True)):
            old_feat = self.bank.get(gid, {}).get("feature")
            if old_feat is not None:
                sim_self = feature_cosine_similarity(feat, old_feat)
                min_self = float(getattr(args, "a_anchor_reid_anchor_update_min_self_sim", 0.50))
                if sim_self < min_self:
                    # A 相机 local_id 偶发跳变或多人粘连时，不让坏特征污染 P 库。
                    return
        if gid not in self.bank:
            self.bank[gid] = {
                "feature": feat.astype(np.float32),
                "count": 1,
                "first_seen_us": int(ts_us),
                "last_seen_us": int(ts_us),
                "source_anchor": str(cam_id),
                "local_ids": [int(local_id)],
            }
            return
        rec = self.bank[gid]
        alpha = float(getattr(args, "a_anchor_reid_feature_ema_alpha", 0.86))
        old = rec.get("feature")
        if old is None:
            new_feat = feat.astype(np.float32)
        else:
            new_feat = _l2_normalize_feature(alpha * np.asarray(old, dtype=np.float32) + (1.0 - alpha) * feat.astype(np.float32))
            if new_feat is None:
                new_feat = feat.astype(np.float32)
        rec["feature"] = new_feat
        rec["count"] = int(rec.get("count", 0)) + 1
        rec["last_seen_us"] = int(ts_us)
        lids = list(rec.get("local_ids") or [])
        if int(local_id) not in lids:
            lids.append(int(local_id))
        rec["local_ids"] = lids[-12:]

    def _assign_person(self, p: PersonInstance, gid: Optional[int], score: float, margin: float, source: str) -> None:
        p.global_id_from_A = int(gid) if gid is not None and int(gid) > 0 else None
        p.reid_score = float(score)
        p.reid_margin = float(margin)
        p.reid_source_anchor = str(source or self.anchor_camera)

    def _reset_anchor_bank_for_group_split(self, args: argparse.Namespace, reason: str = "group_split") -> None:
        """开局多人粘连后分裂：丢弃把多人整体注册出来的旧 P1，重新从分开的人开始建库。"""
        if not bool(getattr(args, "a_anchor_reid_group_guard_reset_after_split", True)):
            return
        print(f"[A-anchor][WARN] reset A identity bank because of {reason}; previous bank_size={len(self.bank)}", flush=True)
        self.bank.clear()
        self.local_to_global.clear()
        self.assignment_cache.clear()
        self.next_global_id = int(getattr(args, "a_anchor_reid_start_id", 1))

    def _process_anchor(self, cam_id: str, persons: List[PersonInstance], frame: np.ndarray, ts_us: int, fid: int, args: argparse.Namespace) -> None:
        if not persons:
            self.last_anchor_single = None
            return
        min_conf = float(getattr(args, "a_anchor_reid_anchor_min_conf", 0.35))
        min_area = int(getattr(args, "a_anchor_reid_anchor_min_area", 220))
        match_thr = float(getattr(args, "a_anchor_reid_anchor_match_thresh", 0.72))
        margin_thr = float(getattr(args, "a_anchor_reid_anchor_min_margin", 0.035))
        frame_shape = frame.shape

        # 1) 开局/遮挡时，A 相机只有一个很宽/很大的 mask，常常是多个人粘成一个人。
        if len(persons) == 1:
            p0 = persons[0]
            group_like = is_anchor_group_candidate(p0, frame_shape, args)
            self.last_anchor_single = {
                "ts_us": int(ts_us),
                "bbox": tuple(int(v) for v in p0.bbox_xyxy),
                "area": int(getattr(p0, "area", 0) or 0),
                "group_like": bool(group_like),
                "stats": anchor_person_shape_stats(p0, frame_shape),
            }
            if group_like and bool(getattr(args, "a_anchor_reid_group_guard_no_enroll_single", True)):
                self._assign_person(p0, None, 0.0, 0.0, "A_GROUP_WAIT")
                return
        else:
            last = self.last_anchor_single if isinstance(self.last_anchor_single, dict) else None
            if last and bool(last.get("group_like", False)):
                age_ms = (int(ts_us) - int(last.get("ts_us", ts_us))) / 1000.0
                split_window_ms = float(getattr(args, "a_anchor_reid_group_guard_split_window_ms", 6000.0))
                reset_limit = int(getattr(args, "a_anchor_reid_group_guard_reset_if_bank_size_le", 2))
                if age_ms <= split_window_ms and int(len(self.bank)) <= reset_limit:
                    self._reset_anchor_bank_for_group_split(args, reason="A_single_group_mask_split_to_multi")
            self.last_anchor_single = None

        used_gids: set = set()
        bank_ids_at_frame_start: set = set(int(g) for g in self.bank.keys())
        pending_create: List[Tuple[int, PersonInstance, Optional[np.ndarray], float, float]] = []
        hsv_holder: Dict[str, Any] = {"hsv": None}
        lowfreq = self._lowfreq_enabled(args)
        anchor_update_every = max(1, int(getattr(args, "a_anchor_reid_anchor_update_every_n", 5)))

        # 第一遍：已有 local_id -> global_id 绑定时，先保持标签；只按低频更新 A 库特征。
        for pi, p in enumerate(persons):
            local_id = self._local_id_of(p)
            key = self._cache_key(cam_id, local_id)
            rec = self.assignment_cache.get(key, {})
            gid = self.local_to_global.get(key)
            if gid is not None and int(gid) in used_gids:
                self._clear_assignment(key)
                gid = None
            if gid is not None and int(gid) in self.bank:
                gid = int(gid)
                used_gids.add(gid)
                score = float(rec.get("score", 1.0) if isinstance(rec, dict) else 1.0)
                margin = float(rec.get("margin", 1.0) if isinstance(rec, dict) else 1.0)
                self._assign_person(p, gid, score, margin, self.anchor_camera)
                self._record_bank_anchor_state(gid, p, fid, ts_us)
                self._remember_assignment(key, gid, score, margin, fid, ts_us, reid_checked=False, status="anchor_cache")

                last_update = int(rec.get("last_update_frame", -10**9)) if isinstance(rec, dict) else -10**9
                due_update = (not lowfreq) or last_update < 0 or (int(fid) - last_update) >= anchor_update_every
                if due_update and float(p.confidence) >= min_conf and int(p.area) >= min_area:
                    feat = self._extract_feature_cached(frame, p, args, hsv_holder)
                    if feat is not None:
                        self._update_bank(gid, feat, cam_id, ts_us, local_id, args)
                        self._record_bank_anchor_state(gid, p, fid, ts_us)
                        self._remember_assignment(key, gid, 1.0, 1.0, fid, ts_us, reid_checked=True, bank_updated=True, status="anchor_update")
                continue
            pending_create.append((pi, p, None, 0.0, 0.0))

        # 第二遍：未绑定的人只允许匹配“本帧开始前已经存在、且本帧尚未使用”的 A 库 ID。
        for pi, p, _, _, _ in pending_create:
            local_id = self._local_id_of(p)
            key = self._cache_key(cam_id, local_id)
            feat = None
            if float(p.confidence) >= min_conf and int(p.area) >= min_area:
                feat = self._extract_feature_cached(frame, p, args, hsv_holder)
            best_gid, best_score, best_margin = self._match_feature(
                feat,
                exclude_gids=used_gids,
                allow_gids=bank_ids_at_frame_start,
            )
            gid: Optional[int] = None
            score, margin = float(best_score), float(best_margin)
            if best_gid is not None and best_score >= match_thr and best_margin >= margin_thr:
                gid = int(best_gid)
            elif int(len(self.bank)) < self.max_ids and float(p.confidence) >= min_conf and int(p.area) >= min_area and feat is not None:
                gid = self._new_global_id()
                score, margin = 1.0, 1.0
            else:
                gid = None

            if gid is not None and int(gid) in used_gids:
                gid = None
            if gid is not None:
                gid = int(gid)
                used_gids.add(gid)
                self.local_to_global[key] = gid
                if float(p.confidence) >= min_conf and int(p.area) >= min_area and feat is not None:
                    self._update_bank(gid, feat, cam_id, ts_us, local_id, args)
                    self._record_bank_anchor_state(gid, p, fid, ts_us)
                    self._remember_assignment(key, gid, score, margin, fid, ts_us, reid_checked=True, bank_updated=True, status="anchor_new_or_match")
                else:
                    self._remember_assignment(key, gid, score, margin, fid, ts_us, reid_checked=False, status="anchor_match")
                self._assign_person(p, gid, score, margin, self.anchor_camera)
            else:
                self._remember_unknown(key, fid, ts_us, score, margin)
                self._assign_person(p, None, score, margin, self.anchor_camera)

    def _process_query_camera(self, cam_id: str, persons: List[PersonInstance], frame: np.ndarray, ts_us: int, fid: int, args: argparse.Namespace) -> None:
        if not persons:
            return
        if not self.bank:
            for p in persons:
                self._assign_person(p, None, 0.0, 0.0, self.anchor_camera)
            return

        min_conf = float(getattr(args, "a_anchor_reid_query_min_conf", 0.18))
        min_area = int(getattr(args, "a_anchor_reid_query_min_area", 80))
        match_thr = float(getattr(args, "a_anchor_reid_query_match_thresh", 0.58))
        margin_thr = float(getattr(args, "a_anchor_reid_query_min_margin", 0.015))
        confirmed_every = max(1, int(getattr(args, "a_anchor_reid_confirmed_recheck_every_n", 10)))
        unknown_every = max(1, int(getattr(args, "a_anchor_reid_unknown_recheck_every_n", 2)))
        switch_thr = float(getattr(args, "a_anchor_reid_switch_match_thresh", max(0.72, match_thr + 0.12)))
        switch_margin = float(getattr(args, "a_anchor_reid_switch_min_margin", max(0.05, margin_thr + 0.03)))
        keep_on_fail = bool(getattr(args, "a_anchor_reid_keep_confirmed_on_recheck_fail", True))
        lowfreq = self._lowfreq_enabled(args)

        # 同一帧如果多个 local_id 缓存到同一个 P，强制这些目标做 ReID 复核，避免同帧重复显示同一个 P。
        cached_counts: Dict[int, int] = {}
        cached_by_pi: Dict[int, Tuple[Tuple[str, int], Dict[str, Any], Optional[int]]] = {}
        for pi, p in enumerate(persons):
            local_id = self._local_id_of(p)
            key = self._cache_key(cam_id, local_id)
            rec = self.assignment_cache.get(key, {})
            gid = self.local_to_global.get(key)
            if gid is not None and int(gid) not in self.bank:
                self._clear_assignment(key)
                gid = None
                rec = {}
            valid_cached = gid is not None and int(gid) in self.bank and self._cache_alive(rec, ts_us, args)
            cached_by_pi[pi] = (key, rec, int(gid) if valid_cached else None)
            if valid_cached:
                cached_counts[int(gid)] = cached_counts.get(int(gid), 0) + 1

        rows: List[Tuple[float, int, int, float, float, str, Tuple[str, int], bool]] = []
        assigned_unknown: set = set()
        hsv_holder: Dict[str, Any] = {"hsv": None}

        for pi, p in enumerate(persons):
            key, rec, cached_gid = cached_by_pi.get(pi, (self._cache_key(cam_id, self._local_id_of(p)), {}, None))
            force_due = cached_gid is not None and cached_counts.get(int(cached_gid), 0) > 1
            last_reid = int(rec.get("last_reid_frame", -10**9)) if isinstance(rec, dict) else -10**9
            last_unknown = last_reid
            if cached_gid is not None:
                due = (not lowfreq) or force_due or last_reid < 0 or (int(fid) - last_reid) >= confirmed_every
            else:
                due = last_unknown < 0 or (int(fid) - last_unknown) >= unknown_every

            # 已确认且未到复核间隔：只沿用 P 标签，不计算外观特征；mask/bbox 仍是当前帧 YOLO 的结果。
            if cached_gid is not None and not due:
                score = float(rec.get("score", 1.0)) if isinstance(rec, dict) else 1.0
                margin = float(rec.get("margin", 1.0)) if isinstance(rec, dict) else 1.0
                rows.append((0.98, pi, int(cached_gid), score, margin, "cache", key, False))
                continue

            # 质量太差的帧不做 ReID；如果已有确认 ID，短时间保持标签。
            if float(p.confidence) < min_conf or int(p.area) < min_area:
                if cached_gid is not None and keep_on_fail:
                    score = float(rec.get("score", 0.0)) if isinstance(rec, dict) else 0.0
                    margin = float(rec.get("margin", 0.0)) if isinstance(rec, dict) else 0.0
                    rows.append((0.70, pi, int(cached_gid), score, margin, "hold_low_quality", key, False))
                else:
                    self._remember_unknown(key, fid, ts_us, 0.0, 0.0)
                    self._assign_person(p, None, 0.0, 0.0, self.anchor_camera)
                    assigned_unknown.add(pi)
                continue

            feat = self._extract_feature_cached(frame, p, args, hsv_holder)
            if feat is None:
                if cached_gid is not None and keep_on_fail:
                    score = float(rec.get("score", 0.0)) if isinstance(rec, dict) else 0.0
                    margin = float(rec.get("margin", 0.0)) if isinstance(rec, dict) else 0.0
                    rows.append((0.68, pi, int(cached_gid), score, margin, "hold_no_feature", key, False))
                else:
                    self._remember_unknown(key, fid, ts_us, 0.0, 0.0)
                    self._assign_person(p, None, 0.0, 0.0, self.anchor_camera)
                    assigned_unknown.add(pi)
                continue

            gid, score, margin = self._match_feature(
                feat,
                query_person=p,
                query_frame_id=int(fid),
                args=args,
                apply_local_xy_gate=True,
            )
            score = float(score)
            margin = float(margin)
            if gid is not None and score >= match_thr and margin >= margin_thr:
                gid = int(gid)
                if cached_gid is not None and gid != int(cached_gid):
                    # 已确认 ID 不因为一次低优势复核就切换，只有很明确时才允许切换，避免 P1/P3 互判。
                    if score >= switch_thr and margin >= switch_margin:
                        rows.append((score, pi, gid, score, margin, "switch", key, True))
                    elif keep_on_fail:
                        rows.append((0.72, pi, int(cached_gid), score, margin, "hold_switch_not_clear", key, True))
                    else:
                        self._remember_unknown(key, fid, ts_us, score, margin)
                        self._assign_person(p, None, score, margin, self.anchor_camera)
                        assigned_unknown.add(pi)
                else:
                    if cached_gid is None and not self._remember_pending_match(key, gid, score, margin, fid, ts_us, args):
                        self._assign_person(p, None, score, margin, "pending_confirm")
                        assigned_unknown.add(pi)
                        continue
                    rows.append((score, pi, gid, score, margin, "match", key, True))
            else:
                if cached_gid is not None and keep_on_fail:
                    rows.append((0.70, pi, int(cached_gid), score, margin, "hold_recheck_fail", key, True))
                else:
                    self._remember_unknown(key, fid, ts_us, score, margin)
                    self._assign_person(p, None, score, margin, self.anchor_camera)
                    assigned_unknown.add(pi)

        # Greedy 同帧唯一分配：同一个 P 编号在同一台查询相机同一帧只能给一个人。
        rows.sort(reverse=True, key=lambda x: x[0])
        used_pi = set()
        used_gid = set()
        for priority, pi, gid, score, margin, source, key, reid_checked in rows:
            if pi in used_pi or gid in used_gid:
                continue
            used_pi.add(pi)
            used_gid.add(gid)
            self._assign_person(persons[pi], gid, score, margin, self.anchor_camera)
            self._remember_assignment(key, gid, score, margin, fid, ts_us, reid_checked=bool(reid_checked), status=str(source))

        for pi, p in enumerate(persons):
            if pi in used_pi or pi in assigned_unknown:
                continue
            # 缓存冲突中没抢到唯一 P 的人，不强行显示旧 P，避免同帧重复 ID。
            key, rec, cached_gid = cached_by_pi.get(pi, (self._cache_key(cam_id, self._local_id_of(p)), {}, None))
            if cached_gid is None:
                self._remember_unknown(key, fid, ts_us, float(getattr(p, "reid_score", 0.0)), float(getattr(p, "reid_margin", 0.0)))
            self._assign_person(p, None, float(getattr(p, "reid_score", 0.0)), float(getattr(p, "reid_margin", 0.0)), self.anchor_camera)

    def assign_batch(self, processed_items: List[Tuple[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace], List[PersonInstance]]]) -> None:
        if not self.enabled:
            return
        # A 先注册 / 更新，再让其他相机查询，这样同一轮 batch 中 A 的新身份能立即被 B/C/D/E 使用。
        for meta, persons in processed_items:
            cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args = meta
            if str(cam_id) == self.anchor_camera:
                self._process_anchor(str(cam_id), persons, frame, int(ts_us), int(fid), args)
        for meta, persons in processed_items:
            cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args = meta
            if str(cam_id) == self.anchor_camera:
                continue
            if self.allow_non_anchor_enroll:
                self._process_anchor(str(cam_id), persons, frame, int(ts_us), int(fid), args)
            else:
                self._process_query_camera(str(cam_id), persons, frame, int(ts_us), int(fid), args)

class StablePlayerTrack:
    def __init__(self, stable_id: int, person: PersonInstance, frame_bgr: np.ndarray,
                 ts_us: int, frame_id: int, args: argparse.Namespace):
        self.stable_id = int(stable_id)
        self.bbox_xyxy = tuple(int(v) for v in person.bbox_xyxy)
        self.mask = person.mask.copy()
        self.center = bbox_center_xyxy(self.bbox_xyxy)
        self.local_xy = tuple(float(v) for v in person.local_xy) if person.local_xy is not None else None
        self.center_velocity = (0.0, 0.0)
        self.local_velocity = (0.0, 0.0)
        self.area = int(max(1, person.area))
        self.hist = compute_color_hist(frame_bgr, person.mask) if bool(args.stable_id_enable_color_hist) else None
        self.frame_h, self.frame_w = frame_bgr.shape[:2]
        self.first_seen_us = int(ts_us)
        self.last_seen_us = int(ts_us)
        self.first_seen_frame = int(frame_id)
        self.last_seen_frame = int(frame_id)
        self.miss_frames = 0
        self.seen_frames = 1
        self.last_debug = "new"

    def update(self, person: PersonInstance, frame_bgr: np.ndarray, ts_us: int, frame_id: int, args: argparse.Namespace) -> None:
        old_center = self.center
        old_local = self.local_xy
        old_frame = int(self.last_seen_frame)
        new_bbox = tuple(int(v) for v in person.bbox_xyxy)
        new_center = bbox_center_xyxy(new_bbox)
        new_local = tuple(float(v) for v in person.local_xy) if person.local_xy is not None else old_local
        dt_frames = max(1.0, float(int(frame_id) - old_frame))
        alpha = float(args.stable_id_velocity_ema_alpha)
        measured_cv = ((new_center[0] - old_center[0]) / dt_frames, (new_center[1] - old_center[1]) / dt_frames)
        self.center_velocity = (
            alpha * float(self.center_velocity[0]) + (1.0 - alpha) * measured_cv[0],
            alpha * float(self.center_velocity[1]) + (1.0 - alpha) * measured_cv[1],
        )
        if old_local is not None and new_local is not None:
            measured_lv = ((new_local[0] - old_local[0]) / dt_frames, (new_local[1] - old_local[1]) / dt_frames)
            self.local_velocity = (
                alpha * float(self.local_velocity[0]) + (1.0 - alpha) * measured_lv[0],
                alpha * float(self.local_velocity[1]) + (1.0 - alpha) * measured_lv[1],
            )
        self.bbox_xyxy = new_bbox
        self.mask = person.mask.copy()
        self.center = new_center
        self.local_xy = new_local
        self.area = int(max(1, person.area))
        self.frame_h, self.frame_w = frame_bgr.shape[:2]
        if bool(args.stable_id_enable_color_hist):
            new_hist = compute_color_hist(frame_bgr, person.mask)
            if new_hist is not None:
                if self.hist is None:
                    self.hist = new_hist
                else:
                    a = float(args.stable_id_appearance_ema_alpha)
                    self.hist = cv2.normalize(a * self.hist + (1.0 - a) * new_hist, None).astype(np.float32)
        self.last_seen_us = int(ts_us)
        self.last_seen_frame = int(frame_id)
        self.miss_frames = 0
        self.seen_frames += 1


def _is_near_image_edge(args: argparse.Namespace,
                        bbox_xyxy: Tuple[int, int, int, int],
                        frame_w: int,
                        frame_h: int) -> bool:
    """判断目标是否靠近“当前相机画面边缘”。

    注意：local_xy 在本版本中是“真实场地坐标”，例如 X=0~31.8、Y=-20.4~0，
    这表示整个场地边界，不等于当前相机可见边界。
    因此 ID 逻辑里用于放宽重现匹配的 edge_related，必须使用图像 bbox 是否靠近
    当前相机画面边缘，而不是用真实场地边界判断。
    """
    if not bool(getattr(args, "stable_id_use_image_edge", True)):
        return False
    if frame_w <= 0 or frame_h <= 0:
        return False
    margin = float(getattr(args, "stable_id_image_edge_margin_px", 70.0))
    if margin <= 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return (x1 <= margin or y1 <= margin or
            x2 >= float(frame_w - 1) - margin or
            y2 >= float(frame_h - 1) - margin)


def _predict_center(tr: StablePlayerTrack, frame_id: int, args: argparse.Namespace) -> Tuple[float, float]:
    if not bool(args.stable_id_motion_predict_enable):
        return tr.center
    dt = max(0.0, float(int(frame_id) - int(tr.last_seen_frame)))
    dt = min(dt, float(args.stable_id_motion_max_predict_frames))
    return (float(tr.center[0]) + float(tr.center_velocity[0]) * dt,
            float(tr.center[1]) + float(tr.center_velocity[1]) * dt)


def _predict_local(tr: StablePlayerTrack, frame_id: int, args: argparse.Namespace) -> Optional[Tuple[float, float]]:
    if tr.local_xy is None:
        return None
    if not bool(args.stable_id_motion_predict_enable):
        return tr.local_xy
    dt = max(0.0, float(int(frame_id) - int(tr.last_seen_frame)))
    dt = min(dt, float(args.stable_id_motion_max_predict_frames))
    return (float(tr.local_xy[0]) + float(tr.local_velocity[0]) * dt,
            float(tr.local_xy[1]) + float(tr.local_velocity[1]) * dt)



def _get_occlusion_zones(args: argparse.Namespace) -> List[Dict[str, Any]]:
    zones = getattr(args, "stable_id_occlusion_zones", [])
    if isinstance(zones, list):
        return [z for z in zones if isinstance(z, dict)]
    return []


def _zone_polygon(zone: Dict[str, Any]) -> Optional[np.ndarray]:
    pts = zone.get("local_polygon", None)
    if pts is None:
        pts = zone.get("polygon", None)
    if pts is None:
        pts = zone.get("points", None)
    try:
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
            return arr.reshape(-1, 1, 2)
    except Exception:
        pass
    return None


def _point_near_zone(args: argparse.Namespace, xy: Optional[Tuple[float, float]], zone: Dict[str, Any]) -> Tuple[bool, float]:
    """判断 local_xy 是否在障碍物区域内或距离障碍物区域足够近。返回 (near, signed_dist)。

    signed_dist >= 0 表示点在多边形内部；<0 表示在外部，绝对值近似为到边界距离。
    """
    if xy is None:
        return False, -1e9
    poly = _zone_polygon(zone)
    if poly is None:
        return False, -1e9
    margin = float(zone.get("margin", getattr(args, "stable_id_occlusion_margin", 8.0)))
    try:
        dist = float(cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), True))
        return dist >= -margin, dist
    except Exception:
        return False, -1e9


def _same_occlusion_zone(args: argparse.Namespace,
                         old_xy: Optional[Tuple[float, float]],
                         new_xy: Optional[Tuple[float, float]]) -> Tuple[bool, str, float, float]:
    """旧位置和新位置是否都位于同一个障碍物遮挡通道附近。"""
    if not bool(getattr(args, "stable_id_occlusion_enable", False)):
        return False, "", -1e9, -1e9
    if old_xy is None or new_xy is None:
        return False, "", -1e9, -1e9
    for idx, zone in enumerate(_get_occlusion_zones(args)):
        near_old, d_old = _point_near_zone(args, old_xy, zone)
        near_new, d_new = _point_near_zone(args, new_xy, zone)
        if near_old and near_new:
            return True, str(zone.get("name", f"zone{idx+1}")), d_old, d_new
    return False, "", -1e9, -1e9


class StrictStableIDManager:
    """单相机严格 ID 管理器。

    原则：
    1. 同一个人短暂漏检后，位置/local_xy 接近才复用旧 ID；
    2. 旧 ID 不是从边界消失，远处新出现的人必须新建 ID；
    3. 不再使用“画面只有一个人就强制复用旧 ID”的逻辑；
    4. 旧 ID 记忆保留，但不是随便复用。这样能避免 P1 丢失后远处新人被标为 P1。
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.next_id = int(args.stable_id_start_id)
        self.tracks: Dict[int, StablePlayerTrack] = {}

    def _new_id(self) -> int:
        sid = int(self.next_id)
        self.next_id += 1
        return sid

    def _prune(self, ts_us: int) -> None:
        if bool(self.args.stable_id_keep_lost_forever):
            return
        inactive_ms = float(self.args.stable_id_inactive_memory_ms)
        remove = []
        for sid, tr in self.tracks.items():
            age_ms = (int(ts_us) - int(tr.last_seen_us)) / 1000.0
            if age_ms > inactive_ms:
                remove.append(sid)
        for sid in remove:
            self.tracks.pop(sid, None)

    def _distances(self, p: PersonInstance, tr: StablePlayerTrack, frame_id: int) -> Tuple[float, float, float, float]:
        pc = bbox_center_xyxy(p.bbox_xyxy)
        pred_c = _predict_center(tr, frame_id, self.args)
        center_now = float(np.hypot(pc[0] - tr.center[0], pc[1] - tr.center[1]))
        center_pred = float(np.hypot(pc[0] - pred_c[0], pc[1] - pred_c[1]))
        center_dist = min(center_now, center_pred)
        local_dist = 1e9
        local_now = 1e9
        if bool(self.args.stable_id_use_local_xy) and p.local_xy is not None and tr.local_xy is not None:
            pred_l = _predict_local(tr, frame_id, self.args)
            local_now = float(np.hypot(float(p.local_xy[0]) - float(tr.local_xy[0]),
                                       float(p.local_xy[1]) - float(tr.local_xy[1])))
            if pred_l is not None:
                local_pred = float(np.hypot(float(p.local_xy[0]) - float(pred_l[0]),
                                            float(p.local_xy[1]) - float(pred_l[1])))
                local_dist = min(local_now, local_pred)
            else:
                local_dist = local_now
        return center_dist, center_now, local_dist, local_now

    def _score(self, p: PersonInstance, p_hist: Optional[np.ndarray], tr: StablePlayerTrack,
               ts_us: int, frame_id: int, frame_shape: Tuple[int, int, int]) -> Tuple[float, Dict[str, Any]]:
        args = self.args
        lost_ms = (int(ts_us) - int(tr.last_seen_us)) / 1000.0
        iou = bbox_iou_xyxy(p.bbox_xyxy, tr.bbox_xyxy)
        contain = bbox_contain_xyxy(p.bbox_xyxy, tr.bbox_xyxy)
        center_dist, center_now, local_dist, local_now = self._distances(p, tr, frame_id)
        sr = size_ratio(int(p.area), int(tr.area))
        app = hist_similarity(p_hist, tr.hist) if bool(args.stable_id_enable_color_hist) else 0.0
        frame_h, frame_w = frame_shape[:2]
        old_edge = _is_near_image_edge(args, tr.bbox_xyxy, int(getattr(tr, "frame_w", frame_w)), int(getattr(tr, "frame_h", frame_h)))
        new_edge = _is_near_image_edge(args, p.bbox_xyxy, int(frame_w), int(frame_h))
        edge_related = bool(old_edge or new_edge)
        occlusion_related, occlusion_name, occ_old_dist, occ_new_dist = _same_occlusion_zone(args, tr.local_xy, p.local_xy)
        recent_ms = float(args.stable_id_strict_reappear_after_ms)

        # 1) 极强重叠/包含通常是同一个目标，直接允许进入打分。
        overlap_ok = iou >= float(args.stable_id_match_iou_thresh) or contain >= float(args.stable_id_match_contain_thresh)

        # 2) 严格防换绑：旧 ID 已经丢失一段时间后，如果不是边界相关，远处新人不能复用旧 ID。
        if bool(args.stable_id_strict_reappear_enable) and lost_ms > recent_ms and not overlap_ok:
            occlusion_usable = (
                bool(occlusion_related)
                and lost_ms <= float(args.stable_id_occlusion_memory_ms)
                and lost_ms >= float(args.stable_id_occlusion_min_lost_ms)
            )
            if occlusion_usable:
                center_limit = float(args.stable_id_occlusion_center_dist_px)
                local_limit = float(args.stable_id_occlusion_local_dist)
                app_thresh = float(args.stable_id_occlusion_appearance_thresh)
            elif edge_related:
                center_limit = float(args.stable_id_edge_reappear_center_dist_px)
                local_limit = float(args.stable_id_edge_reappear_local_dist)
                app_thresh = float(args.stable_id_strict_reappear_appearance_thresh)
            else:
                center_limit = float(args.stable_id_strict_reappear_center_dist_px)
                local_limit = float(args.stable_id_strict_reappear_local_dist)
                app_thresh = float(args.stable_id_strict_reappear_appearance_thresh)

            local_available = local_dist < 1e8
            local_ok = local_available and local_dist <= local_limit
            center_ok = center_dist <= center_limit
            app_ok = app >= app_thresh

            # hard reject：旧 ID 不是从边界/障碍物消失，且出现位置明显太远，判定为新人。
            # 如果配置了同一障碍物遮挡区，允许 P4 从障碍物另一侧重新接回。
            if not edge_related and not occlusion_usable:
                if center_now > float(args.stable_id_new_person_center_jump_px):
                    return 0.0, {"reject": "far_center_reappear", "lost_ms": lost_ms, "center": center_now, "local": local_now, "app": app}
                if local_available and local_now > float(args.stable_id_new_person_local_jump):
                    return 0.0, {"reject": "far_local_reappear", "lost_ms": lost_ms, "center": center_now, "local": local_now, "app": app}

            # 长时间/明显丢失后：普通情况仍严格；障碍物情况允许在同一遮挡区附近跨越。
            reappear_ok = local_ok or center_ok or (app_ok and center_dist <= center_limit * 1.35)
            if occlusion_usable:
                reappear_ok = reappear_ok or (app_ok and local_available and local_dist <= local_limit * 1.30)
            if not reappear_ok:
                return 0.0, {"reject": "strict_reappear_gate", "lost_ms": lost_ms, "center": center_dist, "local": local_dist, "app": app, "occ": occlusion_name}

        # 3) 连续帧/短暂漏检的普通门限。
        occlusion_usable = (
            bool(occlusion_related)
            and lost_ms <= float(args.stable_id_occlusion_memory_ms)
            and lost_ms >= float(args.stable_id_occlusion_min_lost_ms)
        )
        if lost_ms <= recent_ms:
            center_thr = float(args.stable_id_match_center_dist_px)
            local_thr = float(args.stable_id_match_local_dist)
        elif occlusion_usable:
            center_thr = float(args.stable_id_occlusion_center_dist_px)
            local_thr = float(args.stable_id_occlusion_local_dist)
        else:
            center_thr = float(args.stable_id_edge_reappear_center_dist_px if edge_related else args.stable_id_strict_reappear_center_dist_px)
            local_thr = float(args.stable_id_edge_reappear_local_dist if edge_related else args.stable_id_strict_reappear_local_dist)

        if str(getattr(p, "visibility", "unknown")) in {"partial", "small_or_head", "unknown"}:
            center_thr *= float(args.stable_id_partial_gate_boost)
            local_thr *= float(args.stable_id_partial_gate_boost)
        if edge_related:
            center_thr *= float(args.stable_id_edge_gate_boost)
            local_thr *= float(args.stable_id_edge_gate_boost)

        position_ok = center_dist <= center_thr and sr <= float(args.stable_id_match_size_ratio)
        local_ok = local_dist <= local_thr and sr <= float(args.stable_id_match_size_ratio)
        app_gate = float(args.stable_id_occlusion_appearance_thresh) if occlusion_usable else float(args.stable_id_appearance_match_thresh)
        appearance_ok = app >= app_gate and center_dist <= center_thr * 1.5
        valid = bool(overlap_ok or position_ok or local_ok or appearance_ok)
        if not valid:
            return 0.0, {"reject": "no_valid_condition", "lost_ms": lost_ms, "center": center_dist, "local": local_dist, "app": app, "iou": iou}

        pos_score = max(0.0, 1.0 - min(1.0, center_dist / max(1.0, center_thr)))
        local_score = 0.0 if local_dist > 1e8 else max(0.0, 1.0 - min(1.0, local_dist / max(1.0, local_thr)))
        overlap_score = max(iou / max(1e-6, float(args.stable_id_match_iou_thresh)), contain)
        overlap_score = max(0.0, min(1.0, overlap_score))
        size_score = max(0.0, 1.0 - min(1.0, (sr - 1.0) / max(1.0, float(args.stable_id_match_size_ratio) - 1.0)))
        recency_score = max(0.0, 1.0 - min(1.0, lost_ms / max(1.0, float(args.stable_id_reuse_memory_ms))))

        score = (
            float(args.stable_id_position_weight) * pos_score
            + float(args.stable_id_local_weight) * local_score
            + float(args.stable_id_overlap_weight) * overlap_score
            + float(args.stable_id_appearance_weight) * app
            + float(args.stable_id_size_weight) * size_score
            + float(args.stable_id_recency_weight) * recency_score
        )
        if occlusion_usable:
            score += float(args.stable_id_occlusion_score_bonus)
        return float(score), {
            "lost_ms": lost_ms, "center": center_dist, "local": local_dist, "app": app,
            "iou": iou, "contain": contain, "score": score, "edge": 1 if edge_related else 0,
            "occ": occlusion_name if occlusion_usable else "",
        }

    def _track_lost_ms(self, tr: StablePlayerTrack, ts_us: int) -> float:
        return (int(ts_us) - int(tr.last_seen_us)) / 1000.0

    def _is_active_track(self, tr: StablePlayerTrack, ts_us: int) -> bool:
        # 连续可见/短时漏检的轨迹优先匹配。这个优先级用于防止 lost P2 抢走仍在画面里的 P3。
        max_miss = int(getattr(self.args, "stable_id_active_max_miss_frames", 12))
        grace_ms = float(getattr(self.args, "stable_id_active_grace_ms", 1200.0))
        return int(getattr(tr, "miss_frames", 9999)) <= max_miss and self._track_lost_ms(tr, ts_us) <= grace_ms

    def _active_match_score(self,
                            p: PersonInstance,
                            p_hist: Optional[np.ndarray],
                            tr: StablePlayerTrack,
                            ts_us: int,
                            frame_id: int) -> Tuple[float, Dict[str, Any]]:
        """连续轨迹匹配分数。

        这里比 lost_reid 宽松，因为 active 轨迹本来就是连续的人。
        重点看：运动预测位置、local_xy、bbox/mask 重叠、面积变化。
        颜色只作为低权重辅助，不作为硬门槛，避免 mask 抖动导致同一个人换 ID。
        """
        args = self.args
        lost_ms = self._track_lost_ms(tr, ts_us)
        iou = bbox_iou_xyxy(p.bbox_xyxy, tr.bbox_xyxy)
        contain = bbox_contain_xyxy(p.bbox_xyxy, tr.bbox_xyxy)
        center_dist, center_now, local_dist, local_now = self._distances(p, tr, frame_id)
        sr = size_ratio(int(p.area), int(tr.area))
        app = hist_similarity(p_hist, tr.hist) if bool(args.stable_id_enable_color_hist) else 0.0

        center_thr = float(getattr(args, "stable_id_active_center_dist_px", 520.0))
        local_thr = float(getattr(args, "stable_id_active_local_dist", 5.0))
        size_thr = float(getattr(args, "stable_id_active_size_ratio", 12.0))
        if str(getattr(p, "visibility", "unknown")) in {"partial", "small_or_head", "unknown"}:
            boost = float(getattr(args, "stable_id_active_partial_boost", 1.35))
            center_thr *= boost
            local_thr *= boost

        local_available = local_dist < 1e8
        center_ok = center_dist <= center_thr
        local_ok = local_available and local_dist <= local_thr
        overlap_ok = iou >= float(args.stable_id_match_iou_thresh) or contain >= float(args.stable_id_match_contain_thresh)
        size_ok = sr <= size_thr
        if not (size_ok and (center_ok or local_ok or overlap_ok)):
            return 0.0, {"reject": "active_gate", "lost_ms": lost_ms, "center": center_dist, "local": local_dist, "app": app, "iou": iou, "sr": sr}

        pos_score = max(0.0, 1.0 - min(1.0, center_dist / max(1.0, center_thr)))
        local_score = 0.0 if not local_available else max(0.0, 1.0 - min(1.0, local_dist / max(1e-6, local_thr)))
        overlap_score = max(iou / max(1e-6, float(args.stable_id_match_iou_thresh)), contain)
        overlap_score = max(0.0, min(1.0, overlap_score))
        size_score = max(0.0, 1.0 - min(1.0, (sr - 1.0) / max(1.0, size_thr - 1.0)))
        recency_score = max(0.0, 1.0 - min(1.0, lost_ms / max(1.0, float(getattr(args, "stable_id_active_grace_ms", 1200.0)))))
        score = (
            float(getattr(args, "stable_id_active_position_weight", 0.42)) * pos_score
            + float(getattr(args, "stable_id_active_local_weight", 0.32)) * local_score
            + float(getattr(args, "stable_id_active_overlap_weight", 0.18)) * overlap_score
            + float(getattr(args, "stable_id_active_appearance_weight", 0.04)) * app
            + float(getattr(args, "stable_id_active_size_weight", 0.02)) * size_score
            + float(getattr(args, "stable_id_active_recency_weight", 0.02)) * recency_score
        )
        return float(score), {"phase_score": "active", "lost_ms": lost_ms, "center": center_dist, "local": local_dist, "app": app, "iou": iou, "contain": contain, "sr": sr, "score": score}

    def _same_place_hold_score(self,
                               p: PersonInstance,
                               p_hist: Optional[np.ndarray],
                               tr: StablePlayerTrack,
                               ts_us: int,
                               frame_id: int) -> Tuple[float, Dict[str, Any]]:
        """原地遮挡/原地漏检后的 ID 保持通道。

        这一步专门处理用户反馈的情况：玩家在同一个位置被障碍物挡住或 YOLO 漏检，
        下一秒/几秒后又在几乎同一个位置出现。它不是普通 lost_reid，因此不能强制要求
        衣服颜色很像；刚露头、半身、mask 抖动时颜色直方图会变化很大。

        安全原则：
          1) active 已经优先匹配，hold 只能匹配剩余 detection，不能抢正在连续跟踪的人；
          2) 必须和旧位置非常近，优先使用 local_xy 的“实际上一位置”，不用速度预测；
          3) 时间窗口有限，避免另一个人过了很久站到同一位置还继承旧 ID；
          4) 如果位置只是一般接近，则仍然要求较低的外观相似度。
        """
        args = self.args
        if not bool(getattr(args, "stable_id_hold_enable", True)):
            return 0.0, {"reject": "hold_disabled"}

        lost_ms = self._track_lost_ms(tr, ts_us)
        max_ms = float(getattr(args, "stable_id_hold_max_ms", 8000.0))
        if lost_ms > max_ms:
            return 0.0, {"reject": "hold_too_old", "lost_ms": lost_ms}

        # 原地遮挡要看“旧位置”和“新位置”真实距离，不看预测点，避免静止/躲藏时速度预测把人推走。
        pc = bbox_center_xyxy(p.bbox_xyxy)
        center_now = float(np.hypot(pc[0] - tr.center[0], pc[1] - tr.center[1]))
        local_now = 1e9
        local_available = bool(getattr(args, "stable_id_use_local_xy", True) and p.local_xy is not None and tr.local_xy is not None)
        if local_available:
            local_now = float(np.hypot(float(p.local_xy[0]) - float(tr.local_xy[0]),
                                       float(p.local_xy[1]) - float(tr.local_xy[1])))

        iou = bbox_iou_xyxy(p.bbox_xyxy, tr.bbox_xyxy)
        contain = bbox_contain_xyxy(p.bbox_xyxy, tr.bbox_xyxy)
        sr = size_ratio(int(p.area), int(tr.area))
        app = hist_similarity(p_hist, tr.hist) if bool(getattr(args, "stable_id_enable_color_hist", True)) else 1.0

        local_thr = float(getattr(args, "stable_id_hold_local_dist", 3.2))
        center_thr = float(getattr(args, "stable_id_hold_center_dist_px", 360.0))
        tight_local_thr = float(getattr(args, "stable_id_hold_tight_local_dist", 1.4))
        tight_center_thr = float(getattr(args, "stable_id_hold_tight_center_dist_px", 150.0))
        size_thr = float(getattr(args, "stable_id_hold_size_ratio", 14.0))
        app_thr = float(getattr(args, "stable_id_hold_appearance_thresh", 0.30))
        noapp_max_ms = float(getattr(args, "stable_id_hold_noapp_max_ms", 2800.0))

        # v5 新增：多人同时进入障碍物/遮挡区后，可能会在遮挡后交叉。
        # 这时不能只靠“旧位置很近”来接 ID；如果外观足够像，允许在更大的 cross gate 内接回，
        # 后面还会经过 batch 级歧义保护，防止两个人互换 ID。
        cross_enable = bool(getattr(args, "stable_id_hold_cross_enable", True))
        cross_min_lost_ms = float(getattr(args, "stable_id_hold_cross_min_lost_ms", 650.0))
        cross_local_thr = float(getattr(args, "stable_id_hold_cross_local_dist", 7.5))
        cross_center_thr = float(getattr(args, "stable_id_hold_cross_center_dist_px", 900.0))
        cross_app_thr = float(getattr(args, "stable_id_hold_cross_appearance_thresh", 0.58))
        long_ms = float(getattr(args, "stable_id_hold_long_ms", 5000.0))
        long_app_thr = float(getattr(args, "stable_id_hold_long_appearance_thresh", 0.42))

        local_ok = local_available and local_now <= local_thr
        center_ok = center_now <= center_thr
        tight_ok = (local_available and local_now <= tight_local_thr) or center_now <= tight_center_thr
        overlap_ok = iou >= float(args.stable_id_match_iou_thresh) or contain >= float(args.stable_id_match_contain_thresh)
        app_ok = (p_hist is not None and tr.hist is not None and app >= app_thr) or (not bool(getattr(args, "stable_id_enable_color_hist", True)))
        long_app_ok = (p_hist is not None and tr.hist is not None and app >= long_app_thr) or (not bool(getattr(args, "stable_id_enable_color_hist", True)))
        cross_ok = bool(
            cross_enable
            and lost_ms >= cross_min_lost_ms
            and p_hist is not None and tr.hist is not None
            and app >= cross_app_thr
            and ((local_available and local_now <= cross_local_thr) or center_now <= cross_center_thr)
        )
        size_ok = sr <= size_thr

        if not size_ok:
            return 0.0, {"reject": "hold_size", "lost_ms": lost_ms, "sr": sr, "local": local_now, "center": center_now, "app": app}

        # 核心：原地非常接近时，即使颜色暂时不像，也允许接回；否则必须有一定外观相似。
        # v5 额外允许 cross_ok：多人在障碍物后交叉后，不再强行要求回到原消失位置，
        # 但必须有较强外观相似度，并且后续还会做多候选歧义过滤。
        if not (tight_ok or overlap_ok or ((local_ok or center_ok) and app_ok) or cross_ok):
            return 0.0, {"reject": "hold_gate", "lost_ms": lost_ms, "local": local_now, "center": center_now, "app": app, "iou": iou, "contain": contain, "cross": 0}
        if not app_ok and not (tight_ok or overlap_ok or cross_ok) and lost_ms > noapp_max_ms:
            return 0.0, {"reject": "hold_noapp_too_long", "lost_ms": lost_ms, "local": local_now, "center": center_now, "app": app}
        # 长时间隐藏后如果不是非常贴近原位置/明显重叠，就要求至少有中等外观相似度，避免很久以后其他人站到附近继承旧 ID。
        if lost_ms > long_ms and not (tight_ok or overlap_ok) and not long_app_ok:
            return 0.0, {"reject": "hold_long_need_app", "lost_ms": lost_ms, "local": local_now, "center": center_now, "app": app, "app_thr": long_app_thr}

        # 分数偏向位置近、时间短；颜色只是辅助。
        local_score = 0.0 if not local_available else max(0.0, 1.0 - min(1.0, local_now / max(1e-6, local_thr)))
        center_score = max(0.0, 1.0 - min(1.0, center_now / max(1.0, center_thr)))
        overlap_score = max(0.0, min(1.0, max(iou / max(1e-6, float(args.stable_id_match_iou_thresh)), contain)))
        recency_score = max(0.0, 1.0 - min(1.0, lost_ms / max(1.0, max_ms)))
        size_score = max(0.0, 1.0 - min(1.0, (sr - 1.0) / max(1.0, size_thr - 1.0)))
        score = (0.38 * max(local_score, center_score) +
                 0.24 * local_score +
                 0.16 * center_score +
                 0.10 * overlap_score +
                 0.07 * recency_score +
                 0.03 * min(1.0, max(0.0, app)) +
                 0.02 * size_score)
        if cross_ok:
            cross_local_score = 0.0 if not local_available else max(0.0, 1.0 - min(1.0, local_now / max(1e-6, cross_local_thr)))
            cross_center_score = max(0.0, 1.0 - min(1.0, center_now / max(1.0, cross_center_thr)))
            cross_score = (0.42 * min(1.0, max(0.0, app)) +
                           0.28 * max(cross_local_score, cross_center_score) +
                           0.15 * recency_score +
                           0.10 * size_score +
                           0.05 * overlap_score)
            score = max(score, cross_score)
        if tight_ok:
            score += 0.08
        if overlap_ok:
            score += 0.06
        return float(score), {
            "phase_score": "same_place_hold",
            "lost_ms": lost_ms,
            "local": local_now,
            "center": center_now,
            "app": app,
            "iou": iou,
            "contain": contain,
            "sr": sr,
            "tight": 1 if tight_ok else 0,
            "cross": 1 if cross_ok else 0,
            "score": score,
        }

    def _lost_reid_allowed(self,
                           p: PersonInstance,
                           p_hist: Optional[np.ndarray],
                           tr: StablePlayerTrack,
                           dbg: Dict[str, Any],
                           ts_us: int,
                           frame_id: int) -> Tuple[bool, Dict[str, Any]]:
        """lost ID 严格复用门槛。

        解决 P2 消失后，P3 走到 P2 消失位置附近，被误认为 P2 的问题。
        lost ID 只能匹配 active/bridge 没有占用的 detection，并且要同时满足位置和衣服颜色。
        """
        if not bool(getattr(self.args, "stable_id_lost_reid_enable", True)):
            return True, dbg
        lost_ms = self._track_lost_ms(tr, ts_us)
        max_ms = float(getattr(self.args, "stable_id_lost_reid_max_ms", 5000.0))
        if lost_ms > max_ms:
            nd = dict(dbg); nd.update({"reject": "lost_reid_too_old", "lost_ms": lost_ms})
            return False, nd

        center_dist, center_now, local_dist, local_now = self._distances(p, tr, frame_id)
        app = hist_similarity(p_hist, tr.hist) if bool(getattr(self.args, "stable_id_enable_color_hist", True)) else 1.0
        local_thr = float(getattr(self.args, "stable_id_lost_reid_local_dist", 1.35))
        center_thr = float(getattr(self.args, "stable_id_lost_reid_center_dist_px", 230.0))
        app_thr = float(getattr(self.args, "stable_id_lost_reid_appearance_thresh", 0.70))
        require_app = bool(getattr(self.args, "stable_id_lost_reid_require_appearance", True))

        local_available = p.local_xy is not None and tr.local_xy is not None and local_dist < 1e8
        pos_ok = (local_dist <= local_thr) if local_available else (center_dist <= center_thr)
        if not pos_ok:
            nd = dict(dbg); nd.update({"reject": "lost_reid_position", "lost_ms": lost_ms, "local": local_dist, "center": center_dist})
            return False, nd
        if require_app and (p_hist is None or tr.hist is None or app < app_thr):
            nd = dict(dbg); nd.update({"reject": "lost_reid_appearance", "lost_ms": lost_ms, "app": app, "app_thr": app_thr, "local": local_dist, "center": center_dist})
            return False, nd

        nd = dict(dbg); nd.update({"lost_reid": 1, "lost_ms": lost_ms, "app": app, "local": local_dist, "center": center_dist})
        return True, nd

    def _is_duplicate_near_assigned(self,
                                    p: PersonInstance,
                                    assigned_persons: set,
                                    persons: List[PersonInstance]) -> bool:
        """防止同一个人的小碎片在 merge 之后仍残留并新建 ID。"""
        if not bool(getattr(self.args, "stable_id_suppress_duplicate_new", True)):
            return False
        if not assigned_persons:
            return False
        center_thr = float(getattr(self.args, "stable_id_duplicate_center_dist_px", 150.0))
        local_thr = float(getattr(self.args, "stable_id_duplicate_local_dist", 1.6))
        contain_thr = float(getattr(self.args, "stable_id_duplicate_contain_thresh", 0.18))
        area_ratio_thr = float(getattr(self.args, "stable_id_duplicate_area_ratio", 0.70))
        pc = bbox_center_xyxy(p.bbox_xyxy)
        for api in assigned_persons:
            ap = persons[api]
            ac = bbox_center_xyxy(ap.bbox_xyxy)
            center_dist = float(np.hypot(pc[0] - ac[0], pc[1] - ac[1]))
            contain = bbox_contain_xyxy(p.bbox_xyxy, ap.bbox_xyxy)
            local_ok = False
            if p.local_xy is not None and ap.local_xy is not None:
                local_dist = float(np.hypot(float(p.local_xy[0]) - float(ap.local_xy[0]), float(p.local_xy[1]) - float(ap.local_xy[1])))
                local_ok = local_dist <= local_thr
            small_enough = int(p.area) <= int(ap.area) * area_ratio_thr
            if small_enough and (center_dist <= center_thr or contain >= contain_thr or local_ok):
                return True
        return False

    def _filter_ambiguous_hold_rows(self,
                                    rows: List[Tuple[float, int, int, Dict[str, Any]]]) -> List[Tuple[float, int, int, Dict[str, Any]]]:
        """v5 多人遮挡歧义保护。

        原 v4 的 same_place_hold 对“单人原地遮挡”效果很好，但多人同时消失、在障碍物后交叉后，
        仅靠旧位置会把 ID 接到错误的人身上。这里在 hold 阶段做 batch 级过滤：
          - 如果某个 detection 同时像多个旧 ID，或者某个旧 ID 同时像多个 detection，要求分数/外观有明确优势；
          - 外观优势不明显时，不把旧 ID 硬接过去，避免直接互换 ID。
        注意：如果两个人衣服高度相似且遮挡过程中完全不可见，单相机无法从物理信息上判断谁是谁，
        此时宁愿不做高风险旧 ID 复用，也不要把 P1/P2 直接互换。
        """
        args = self.args
        if not bool(getattr(args, "stable_id_hold_ambiguous_guard_enable", True)) or len(rows) <= 1:
            return rows

        score_margin_thr = float(getattr(args, "stable_id_hold_min_score_margin", 0.055))
        app_thr = float(getattr(args, "stable_id_hold_ambiguous_app_thresh", 0.50))
        app_margin_thr = float(getattr(args, "stable_id_hold_ambiguous_app_margin", 0.08))
        min_lost_ms = float(getattr(args, "stable_id_hold_ambiguous_min_lost_ms", 650.0))

        by_person: Dict[int, List[Tuple[float, int, int, Dict[str, Any]]]] = {}
        by_track: Dict[int, List[Tuple[float, int, int, Dict[str, Any]]]] = {}
        for r in rows:
            score, pi, sid, dbg = r
            by_person.setdefault(pi, []).append(r)
            by_track.setdefault(sid, []).append(r)

        def second_score(items: List[Tuple[float, int, int, Dict[str, Any]]], current: Tuple[float, int, int, Dict[str, Any]]) -> float:
            vals = [float(x[0]) for x in items if x is not current]
            return max(vals) if vals else -1e9

        def second_app(items: List[Tuple[float, int, int, Dict[str, Any]]], current: Tuple[float, int, int, Dict[str, Any]]) -> float:
            vals = [float(x[3].get("app", 0.0)) for x in items if x is not current]
            return max(vals) if vals else -1e9

        filtered: List[Tuple[float, int, int, Dict[str, Any]]] = []
        for r in rows:
            score, pi, sid, dbg = r
            lost_ms = float(dbg.get("lost_ms", 0.0))
            # 刚刚短时闪断不按多人遮挡处理，保留 v4 的灵敏接回能力。
            if lost_ms < min_lost_ms:
                filtered.append(r)
                continue

            person_compete = len(by_person.get(pi, [])) > 1
            track_compete = len(by_track.get(sid, [])) > 1
            if not (person_compete or track_compete):
                filtered.append(r)
                continue

            p_margin = float(score) - second_score(by_person.get(pi, []), r)
            t_margin = float(score) - second_score(by_track.get(sid, []), r)
            p_app_margin = float(dbg.get("app", 0.0)) - second_app(by_person.get(pi, []), r)
            t_app_margin = float(dbg.get("app", 0.0)) - second_app(by_track.get(sid, []), r)
            app = float(dbg.get("app", 0.0))
            cross = int(dbg.get("cross", 0)) == 1

            # 同时满足“分数有优势”或“外观有优势”才允许旧 ID 接回。
            score_clear = (p_margin >= score_margin_thr if person_compete else True) and (t_margin >= score_margin_thr if track_compete else True)
            app_clear = (app >= app_thr and
                         (p_app_margin >= app_margin_thr if person_compete else True) and
                         (t_app_margin >= app_margin_thr if track_compete else True))

            if score_clear or app_clear:
                filtered.append(r)
            else:
                nd = dict(dbg)
                nd.update({
                    "reject": "hold_ambiguous",
                    "p_margin": round(p_margin, 4),
                    "t_margin": round(t_margin, 4),
                    "p_app_margin": round(p_app_margin, 4),
                    "t_app_margin": round(t_app_margin, 4),
                    "cross": 1 if cross else 0,
                })
                # 不输出这条候选；后续可能由更清晰的候选接回，或者作为新人处理。
                continue
        return filtered

    def update(self, persons: List[PersonInstance], frame_bgr: np.ndarray, ts_us: int, frame_id: int) -> List[PersonInstance]:
        if not bool(self.args.stable_id_enable):
            return persons
        self._prune(ts_us)
        for tr in self.tracks.values():
            tr.miss_frames += 1
        if not persons:
            return []

        person_hists = [compute_color_hist(frame_bgr, p.mask) if bool(self.args.stable_id_enable_color_hist) else None for p in persons]
        reuse_ms = float(self.args.stable_id_reuse_memory_ms)
        track_ids = [sid for sid in self.tracks.keys() if self._track_lost_ms(self.tracks[sid], ts_us) <= reuse_ms]

        active_track_ids: List[int] = []
        lost_track_ids: List[int] = []
        for sid in track_ids:
            if self._is_active_track(self.tracks[sid], ts_us):
                active_track_ids.append(sid)
            else:
                lost_track_ids.append(sid)

        assigned_persons = set()
        assigned_tracks = set()
        person_to_sid: Dict[int, int] = {}
        debug_match: Dict[int, Dict[str, Any]] = {}

        def greedy_assign(rows: List[Tuple[float, int, int, Dict[str, Any]]], phase: str) -> None:
            rows.sort(key=lambda x: x[0], reverse=True)
            for score, pi, sid, dbg in rows:
                if pi in assigned_persons or sid in assigned_tracks:
                    continue
                person_to_sid[pi] = sid
                nd = dict(dbg)
                nd["phase"] = phase
                nd["final_score"] = score
                debug_match[pi] = nd
                assigned_persons.add(pi)
                assigned_tracks.add(sid)

        # 1) active 轨迹优先：连续在画面中的人先保持自己的 ID。
        active_rows: List[Tuple[float, int, int, Dict[str, Any]]] = []
        active_thresh = float(getattr(self.args, "stable_id_active_match_score_thresh", 0.14))
        for pi, p in enumerate(persons):
            for sid in active_track_ids:
                score, dbg = self._active_match_score(p, person_hists[pi], self.tracks[sid], ts_us, frame_id)
                if score >= active_thresh:
                    active_rows.append((score, pi, sid, dbg))
        greedy_assign(active_rows, "active")

        # 1.5) same_place_hold：原地遮挡/原地漏检后恢复原 ID。
        # 这个阶段比 lost_reid 更早，专门解决“同一位置消失又同一位置出现却变新 ID”的问题。
        hold_rows: List[Tuple[float, int, int, Dict[str, Any]]] = []
        hold_max_ms = float(getattr(self.args, "stable_id_hold_max_ms", 8000.0))
        hold_thresh = float(getattr(self.args, "stable_id_hold_match_score_thresh", 0.18))
        hold_track_ids = [sid for sid in track_ids
                          if sid not in assigned_tracks and self._track_lost_ms(self.tracks[sid], ts_us) <= hold_max_ms]
        for pi, p in enumerate(persons):
            if pi in assigned_persons:
                continue
            for sid in hold_track_ids:
                if sid in assigned_tracks:
                    continue
                score, dbg = self._same_place_hold_score(p, person_hists[pi], self.tracks[sid], ts_us, frame_id)
                if score >= hold_thresh:
                    hold_rows.append((score, pi, sid, dbg))
        hold_rows = self._filter_ambiguous_hold_rows(hold_rows)
        greedy_assign(hold_rows, "same_place_hold")

        # 2) bridge：短时漏检但还没完全 lost 的轨迹，用运动预测接回。
        bridge_rows: List[Tuple[float, int, int, Dict[str, Any]]] = []
        bridge_max_ms = float(getattr(self.args, "stable_id_bridge_max_ms", 1500.0))
        bridge_thresh = float(getattr(self.args, "stable_id_bridge_match_score_thresh", 0.12))
        bridge_track_ids = [sid for sid in lost_track_ids if self._track_lost_ms(self.tracks[sid], ts_us) <= bridge_max_ms]
        for pi, p in enumerate(persons):
            if pi in assigned_persons:
                continue
            for sid in bridge_track_ids:
                if sid in assigned_tracks:
                    continue
                score, dbg = self._active_match_score(p, person_hists[pi], self.tracks[sid], ts_us, frame_id)
                if score >= bridge_thresh:
                    bridge_rows.append((score, pi, sid, dbg))
        greedy_assign(bridge_rows, "bridge")

        # 3) lost_reid：只有剩余目标才允许恢复旧 ID，并且必须通过外观/位置门槛。
        lost_rows: List[Tuple[float, int, int, Dict[str, Any]]] = []
        lost_penalty = float(getattr(self.args, "stable_id_lost_track_score_penalty", 0.12))
        for pi, p in enumerate(persons):
            if pi in assigned_persons:
                continue
            for sid in lost_track_ids:
                if sid in assigned_tracks:
                    continue
                tr = self.tracks[sid]
                score, dbg = self._score(p, person_hists[pi], tr, ts_us, frame_id, frame_bgr.shape)
                if score < float(self.args.stable_id_match_score_thresh):
                    continue
                ok, dbg2 = self._lost_reid_allowed(p, person_hists[pi], tr, dbg, ts_us, frame_id)
                if not ok:
                    continue
                lost_rows.append((max(0.0, score - lost_penalty), pi, sid, dbg2))
        greedy_assign(lost_rows, "lost_reid")

        # 4) 未匹配目标：真正独立的人才新建 ID；靠近已有人的小碎片不新建。
        output_persons: List[PersonInstance] = []
        new_min_area = int(getattr(self.args, "stable_id_new_track_min_area", 220))
        for pi, p in enumerate(persons):
            if pi in person_to_sid:
                sid = person_to_sid[pi]
                tr = self.tracks[sid]
                tr.last_debug = "match " + json.dumps(debug_match.get(pi, {}), ensure_ascii=False)[:220]
                tr.update(p, frame_bgr, ts_us, frame_id, self.args)
                p.instance_id = int(sid)
                p.local_stable_id = int(sid)
                output_persons.append(p)
            else:
                if int(p.area) < new_min_area:
                    continue
                if self._is_duplicate_near_assigned(p, assigned_persons, persons):
                    continue
                sid = self._new_id()
                tr = StablePlayerTrack(sid, p, frame_bgr, ts_us, frame_id, self.args)
                tr.last_debug = "new_id"
                self.tracks[sid] = tr
                p.instance_id = int(sid)
                p.local_stable_id = int(sid)
                output_persons.append(p)

        output_persons.sort(key=lambda x: int(x.instance_id))
        return output_persons


class LocalMapProjector:
    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(args.local_map_enable)
        self.H: Optional[np.ndarray] = None
        self.image_points: List[List[float]] = []
        self.local_points: List[List[float]] = []
        self.path = str(args.local_calib_path or "")
        if self.enabled:
            self.load(args)

    def load(self, args: argparse.Namespace) -> None:
        data = None
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[local_map] loaded calibration: {self.path}")
            except Exception as e:
                print(f"[local_map][WARN] failed to load {self.path}: {e}")
        if data is None:
            pts = getattr(args, "local_image_points", []) or []
            lpts = getattr(args, "local_points", []) or []
            if pts and lpts:
                data = {"image_points": pts, "local_points": lpts}
        if not isinstance(data, dict):
            print("[local_map][WARN] no local calibration. local_xy disabled.")
            self.enabled = False
            return
        self.image_points = data.get("image_points", []) or []
        # 兼容两种写法：
        # - local_points：沿用旧字段名，但现在可以直接填真实场地坐标；
        # - field_points：更直观的真实场地坐标字段名。
        self.local_points = data.get("field_points", None) or data.get("local_points", []) or []
        if len(self.image_points) < 4 or len(self.local_points) < 4:
            print("[local_map][WARN] local calibration needs 4 point pairs. local_xy disabled.")
            self.enabled = False
            return
        src = np.asarray(self.image_points, dtype=np.float32)
        dst = np.asarray(self.local_points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src[:4], dst[:4]) if len(src) == 4 else cv2.findHomography(src, dst, cv2.RANSAC, 3.0)[0]
        if self.H is None:
            print("[local_map][WARN] findHomography failed. local_xy disabled.")
            self.enabled = False
        else:
            print(f"[local_map] enabled. points={len(src)}")

    def project(self, pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if not self.enabled or self.H is None or pt is None:
            return None
        p = np.asarray([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.H)[0, 0]
        return (float(q[0]), float(q[1]))


def _visibility_from_bbox(bbox_xyxy: Tuple[int, int, int, int], args: argparse.Namespace) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h = max(1, y2 - y1 + 1)
    if h >= int(args.full_body_min_h_px):
        return "full_or_near"
    if h >= int(args.partial_body_min_h_px):
        return "partial"
    return "small_or_head"


def _camera_center_weight(pt: Tuple[float, float], frame_shape: Tuple[int, int], args: argparse.Namespace) -> float:
    if not bool(getattr(args, "camera_center_weight_enable", True)):
        return 1.0
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if w <= 1 or h <= 1:
        return 1.0
    cx, cy = float(w - 1) * 0.5, float(h - 1) * 0.5
    dx = abs(float(pt[0]) - cx) / max(1.0, cx)
    dy = abs(float(pt[1]) - cy) / max(1.0, cy)
    edge = max(0.0, min(1.0, float(np.hypot(dx, dy) / np.sqrt(2.0))))
    power = max(0.25, float(getattr(args, "camera_center_weight_power", 2.0)))
    min_w = max(0.0, min(1.0, float(getattr(args, "camera_center_min_weight", 0.15))))
    return float(min_w + (1.0 - min_w) * ((1.0 - edge) ** power))


def _topdown_anchor_pixel(mask_u8: np.ndarray,
                          bbox_xyxy: Tuple[int, int, int, int],
                          args: argparse.Namespace) -> Tuple[int, int, str]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 5:
        return (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5), "bbox_center")
    method = str(getattr(args, "topdown_anchor_method", "mask_distance_center") or "mask_distance_center").lower()
    work = (mask_u8 > 0).astype(np.uint8) * 255
    erode_px = max(0, int(getattr(args, "topdown_erode_px", 2)))
    if erode_px > 0:
        k = erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(work, kernel, iterations=1)
        if int(np.count_nonzero(eroded)) >= max(8, int(0.20 * len(xs))):
            work = eroded
            ys, xs = np.where(work > 0)
    if method in {"mask_distance_center", "distance_center", "dist_center"}:
        try:
            dist = cv2.distanceTransform((work > 0).astype(np.uint8), cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            if float(max_val) > 0.0:
                return (int(max_loc[0]), int(max_loc[1]), "mask_distance_center")
        except Exception:
            pass
    if method in {"mask_distance_center", "distance_center", "dist_center", "mask_centroid", "centroid"} and len(xs) >= 5:
        return (int(round(float(np.mean(xs)))), int(round(float(np.mean(ys)))), "mask_centroid")
    return (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5), "bbox_center")


def compute_foot_pixel(mask_u8: np.ndarray,
                       bbox_xyxy: Tuple[int, int, int, int],
                       args: argparse.Namespace,
                       frame_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int, str, str, float]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    mode = str(getattr(args, "position_anchor_mode", "topdown_center") or "topdown_center").lower()
    if mode in {"topdown", "topdown_center", "center", "mask_center"}:
        fx, fy, anchor_mode = _topdown_anchor_pixel(mask_u8, bbox_xyxy, args)
        vis = _visibility_from_bbox(bbox_xyxy, args)
        weight = _camera_center_weight((fx, fy), frame_shape or mask_u8.shape[:2], args)
        return int(fx), int(fy), vis, str(anchor_mode), float(weight)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 5:
        fx, fy = int((x1 + x2) * 0.5), int(y2)
        return fx, fy, "bbox_bottom", "foot_bottom", _camera_center_weight((fx, fy), frame_shape or mask_u8.shape[:2], args)
    max_y = int(ys.max())
    band_px = max(3, int(args.foot_band_px))
    band = ys >= max_y - band_px
    xs_band = xs[band]
    fx = int(np.median(xs_band)) if len(xs_band) >= 3 else int(np.median(xs))
    fy = max_y
    vis = _visibility_from_bbox(bbox_xyxy, args)
    return fx, fy, vis, "foot_bottom", _camera_center_weight((fx, fy), frame_shape or mask_u8.shape[:2], args)


def assign_local_positions(persons: List[PersonInstance], projector: Optional[LocalMapProjector], args: argparse.Namespace) -> None:
    for p in persons:
        fx, fy, vis, anchor_mode, center_weight = compute_foot_pixel(p.mask, p.bbox_xyxy, args, p.mask.shape[:2])
        p.foot_pixel = (int(fx), int(fy))
        p.visibility = vis
        p.anchor_pixel_mode = str(anchor_mode)
        p.camera_center_weight = float(center_weight)
        p.coord_quality = "no_projector"
        if projector is not None and projector.enabled:
            p.local_xy = projector.project(p.foot_pixel)
            p.local_xy_raw = p.local_xy
            p.fused_xy = p.local_xy
            p.coord_quality = "good" if p.local_xy is not None else "project_failed"


def apply_topdown_global_person_fusion(
    processed_items: List[Tuple[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace], List[PersonInstance]]]
) -> Dict[str, Any]:
    if not processed_items:
        return {"enabled": False, "reason": "empty"}
    args0 = processed_items[0][0][6]
    if not bool(getattr(args0, "global_person_fusion_enable", True)):
        return {"enabled": False, "reason": "disabled"}

    obs: List[Dict[str, Any]] = []
    for meta, persons in processed_items:
        cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args = meta
        for idx, p in enumerate(persons):
            xy = getattr(p, "local_xy_raw", None) or getattr(p, "local_xy", None)
            if xy is None:
                continue
            try:
                x, y = float(xy[0]), float(xy[1])
            except Exception:
                continue
            if not np.all(np.isfinite((x, y))):
                continue
            obs.append({
                "cam": str(cam_id),
                "idx": int(idx),
                "person": p,
                "x": x,
                "y": y,
                "weight": max(0.01, float(getattr(p, "camera_center_weight", 1.0))),
                "confidence": max(0.01, float(getattr(p, "confidence", 0.0))),
            })
    if not obs:
        return {"enabled": True, "clusters": 0, "observations": 0}

    gate = max(0.05, float(getattr(args0, "global_person_fusion_cluster_dist_m", 0.8)))
    outlier_gate = max(gate, float(getattr(args0, "global_person_fusion_outlier_dist_m", 1.5)))
    clusters: List[List[Dict[str, Any]]] = []
    for item in sorted(obs, key=lambda d: -(float(d["weight"]) * 0.7 + float(d["confidence"]) * 0.3)):
        best_i = -1
        best_d = 1e9
        for ci, members in enumerate(clusters):
            weights = np.asarray([float(m["weight"]) * max(0.05, float(m["confidence"])) for m in members], dtype=np.float64)
            xs = np.asarray([float(m["x"]) for m in members], dtype=np.float64)
            ys = np.asarray([float(m["y"]) for m in members], dtype=np.float64)
            cx = float(np.average(xs, weights=weights))
            cy = float(np.average(ys, weights=weights))
            dd = float(np.hypot(float(item["x"]) - cx, float(item["y"]) - cy))
            if dd < best_d:
                best_d, best_i = dd, ci
        if best_i >= 0 and best_d <= gate:
            clusters[best_i].append(item)
        else:
            clusters.append([item])

    outliers = 0
    for cid, members in enumerate(clusters, start=1):
        if not members:
            continue
        ranked = sorted(members, key=lambda d: float(d["weight"]) * 0.75 + float(d["confidence"]) * 0.25, reverse=True)
        source = ranked[0]
        sx, sy = float(source["x"]), float(source["y"])
        inliers = [m for m in members if float(np.hypot(float(m["x"]) - sx, float(m["y"]) - sy)) <= outlier_gate]
        if not inliers:
            inliers = [source]
        outliers += max(0, len(members) - len(inliers))

        mode = str(getattr(args0, "global_person_fusion_mode", "best_center_weight") or "best_center_weight").lower()
        if mode in {"weighted_average", "weighted", "average"} and len(inliers) >= 2:
            weights = np.asarray([float(m["weight"]) * max(0.05, float(m["confidence"])) for m in inliers], dtype=np.float64)
            fx = float(np.average(np.asarray([float(m["x"]) for m in inliers], dtype=np.float64), weights=weights))
            fy = float(np.average(np.asarray([float(m["y"]) for m in inliers], dtype=np.float64), weights=weights))
        else:
            fx, fy = sx, sy
        source_cams = sorted({str(m["cam"]) for m in inliers})
        state = "confirmed" if len(source_cams) >= 2 else "single_cam"
        for m in members:
            p = m["person"]
            is_inlier = m in inliers
            p.fusion_cluster_id = int(cid)
            p.fusion_state = state if is_inlier else "outlier"
            p.fusion_source_cam = str(source["cam"])
            p.fusion_source_cams = list(source_cams)
            p.fusion_center_weight = float(source["weight"])
            p.fusion_outlier = not bool(is_inlier)
            if is_inlier:
                p.fused_xy = (float(fx), float(fy))
                p.local_xy = p.fused_xy
                p.coord_quality = f"fused_{state}"
            else:
                p.coord_quality = "fusion_outlier"
    return {
        "enabled": True,
        "observations": int(len(obs)),
        "clusters": int(len(clusters)),
        "outliers": int(outliers),
        "cluster_gate_m": float(gate),
        "outlier_gate_m": float(outlier_gate),
    }


def extract_person_instances(result: Any, frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> List[PersonInstance]:
    h, w = frame_shape[:2]
    if result is None or result.masks is None or result.boxes is None:
        return []
    try:
        masks_np = result.masks.data.detach().cpu().numpy()
        cls_np = result.boxes.cls.detach().cpu().numpy()
        conf_np = result.boxes.conf.detach().cpu().numpy()
        xyxy_np = result.boxes.xyxy.detach().cpu().numpy()
    except Exception:
        masks_np = np.asarray(result.masks.data)
        cls_np = np.asarray(result.boxes.cls)
        conf_np = np.asarray(result.boxes.conf)
        xyxy_np = np.asarray(result.boxes.xyxy)

    persons: List[PersonInstance] = []
    n = min(len(masks_np), len(cls_np), len(conf_np), len(xyxy_np))
    for i in range(n):
        if int(cls_np[i]) != int(args.person_class_id):
            continue
        conf = float(conf_np[i])
        if conf < float(args.conf):
            continue
        mask = (masks_np[i] >= float(args.mask_thresh)).astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = postprocess_mask(mask, args)
        for part_idx, part_mask in enumerate(split_person_mask_components(mask, args)):
            area = int(np.count_nonzero(part_mask))
            if area < int(args.min_mask_area):
                continue
            contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) >= float(args.min_contour_area)]
            if not contours:
                continue
            polygons = contours_to_polygons(contours, args.poly_epsilon_ratio)
            if not polygons:
                continue
            ys, xs = np.where(part_mask > 0)
            if len(xs) == 0:
                continue
            bbox = (
                int(max(0, min(w - 1, xs.min()))),
                int(max(0, min(h - 1, ys.min()))),
                int(max(0, min(w - 1, xs.max()))),
                int(max(0, min(h - 1, ys.max()))),
            )
            raw_id = len(persons) + 1
            # source_ids keeps the original YOLO index and split index for debugging
            # while stable_id later assigns the real displayed ID.
            persons.append(PersonInstance(raw_id, conf, bbox, part_mask, contours, polygons, area, [int(i) + 1, int(part_idx) + 1]))
    if bool(args.merge_same_person):
        persons = merge_same_person_instances(persons, args)
    persons.sort(key=lambda p: p.area, reverse=True)
    if int(args.max_persons) > 0:
        persons = persons[: int(args.max_persons)]
    return persons


def infer_persons(model: Any, frame: np.ndarray, args: argparse.Namespace) -> Tuple[List[PersonInstance], float]:
    t1 = time.time()
    kwargs = _build_yolo_predict_kwargs(frame, args, device=getattr(args, "device", ""))
    results = model.predict(**kwargs)
    result = results[0] if results else None
    persons = extract_person_instances(result, frame.shape, args)
    infer_ms = (time.time() - t1) * 1000.0
    return persons, infer_ms


class BaseFrameSource:
    source_name = "base"
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        raise NotImplementedError
    def release(self) -> None:
        pass


class VideoFileSource(BaseFrameSource):
    def __init__(self, path: str, backend: str = "auto"):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.source_name = os.path.basename(self.path)
        self.backend = str(backend or "auto").lower()
        self.cap = None
        self.proc = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.frame_idx = 0
        self.first_frame: Optional[np.ndarray] = None
        self.start_us = now_us()
        if not os.path.exists(self.path):
            raise RuntimeError(f"视频文件不存在: {self.path}")
        if self.backend in {"auto", "opencv"}:
            ok = self._open_opencv()
            if ok:
                self.backend = "opencv"
                print(f"[video] backend=opencv path={self.path}")
                return
            if self.backend == "opencv":
                raise RuntimeError(f"OpenCV 无法打开视频文件: {self.path}")
            print(f"[video][WARN] OpenCV 读取失败，切换 ffmpeg: {self.path}")
        self._open_ffmpeg()
        self.backend = "ffmpeg"
        print(f"[video] backend=ffmpeg size={self.width}x{self.height} fps={self.fps:.3f} path={self.path}")

    def _open_opencv(self) -> bool:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False
        self.first_frame = frame
        self.cap = cap
        self.width = int(frame.shape[1])
        self.height = int(frame.shape[0])
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        self.fps = fps if fps > 0 else 25.0
        return True

    @staticmethod
    def _parse_rate(s: str) -> float:
        try:
            if "/" in s:
                a, b = s.split("/", 1)
                return float(a) / max(1e-9, float(b))
            return float(s)
        except Exception:
            return 25.0

    def _probe_video(self) -> Tuple[int, int, float]:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError("需要 ffprobe/ffmpeg 才能读取该视频。请安装：sudo apt install -y ffmpeg")
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", self.path]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffprobe 读取视频信息失败: {self.path}\n{p.stderr.strip()}")
        data = json.loads(p.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            raise RuntimeError(f"ffprobe 没有找到视频流: {self.path}")
        st = streams[0]
        w, h = int(st.get("width", 0)), int(st.get("height", 0))
        fps = self._parse_rate(st.get("avg_frame_rate") or st.get("r_frame_rate") or "25")
        if w <= 0 or h <= 0:
            raise RuntimeError(f"视频尺寸无效: {w}x{h}")
        return w, h, fps if fps > 0 else 25.0

    def _open_ffmpeg(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("OpenCV 读取失败且系统没有 ffmpeg。请安装：sudo apt install -y ffmpeg")
        self.width, self.height, self.fps = self._probe_video()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", self.path, "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if self.backend == "opencv":
            if self.first_frame is not None:
                frame = self.first_frame
                self.first_frame = None
                self.frame_idx += 1
                return True, frame, self.start_us
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return False, None, now_us()
            pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            ts = self.start_us + int(pos_ms * 1000.0) if pos_ms > 0 else now_us()
            self.frame_idx += 1
            return True, frame, ts
        if self.proc is None or self.proc.stdout is None:
            return False, None, now_us()
        need = self.width * self.height * 3
        buf = self.proc.stdout.read(need)
        if len(buf) != need:
            return False, None, now_us()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        ts = self.start_us + int(self.frame_idx * (1_000_000.0 / max(1e-6, self.fps)))
        self.frame_idx += 1
        return True, frame, ts

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass


class HikMvsSource(BaseFrameSource):
    def __init__(self, serial: str = "", mvs_sdk_path: str = "/opt/MVS/Samples/64/Python/MvImport",
                 timeout_ms: int = 1000, width: int = 0, height: int = 0, fps: float = 0.0):
        self.serial = str(serial or "").strip()
        self.mvs_sdk_path = mvs_sdk_path
        self.timeout_ms = int(timeout_ms)
        self.source_name = f"hik_{self.serial or 'auto'}"
        self.cam = None
        self.mvs = None
        self._import_mvs()
        self._open_device()
        self._configure(width=width, height=height, fps=fps)

    def _import_mvs(self) -> None:
        if self.mvs_sdk_path and self.mvs_sdk_path not in sys.path:
            sys.path.append(self.mvs_sdk_path)
        try:
            self.mvs = importlib.import_module("MvCameraControl_class")
        except Exception as e:
            raise RuntimeError(f"导入海康 MVS Python wrapper 失败：{self.mvs_sdk_path}/MvCameraControl_class.py\n原始错误: {e}")

    @staticmethod
    def _decode_c_array(v: Any) -> str:
        try:
            b = bytes(v)
        except Exception:
            try:
                b = bytes(bytearray(v))
            except Exception:
                return ""
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

    def _device_serial(self, info: Any) -> str:
        mvs = self.mvs
        try:
            if info.nTLayerType == getattr(mvs, "MV_GIGE_DEVICE"):
                return self._decode_c_array(info.SpecialInfo.stGigEInfo.chSerialNumber)
            if info.nTLayerType == getattr(mvs, "MV_USB_DEVICE"):
                return self._decode_c_array(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        except Exception:
            pass
        return ""

    def _open_device(self) -> None:
        mvs = self.mvs
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        tlayer_type = getattr(mvs, "MV_GIGE_DEVICE") | getattr(mvs, "MV_USB_DEVICE")
        ret = mvs.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0 or int(device_list.nDeviceNum) <= 0:
            raise RuntimeError(f"未枚举到海康工业相机设备，ret=0x{ret:x}")
        selected_idx = -1
        found: List[Tuple[int, str]] = []
        for i in range(int(device_list.nDeviceNum)):
            info = ctypes.cast(device_list.pDeviceInfo[i], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            sn = self._device_serial(info)
            found.append((i, sn))
            if self.serial and sn == self.serial:
                selected_idx = i
        if selected_idx < 0:
            if self.serial:
                raise RuntimeError(f"没有找到指定序列号相机: {self.serial}，当前枚举到: {found}")
            selected_idx = 0
        info = ctypes.cast(device_list.pDeviceInfo[selected_idx], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
        sn = self._device_serial(info)
        self.source_name = f"hik_{sn or selected_idx}"
        print(f"[HIK] selected index={selected_idx}, serial={sn or '(unknown)'}")
        self.cam = mvs.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:x}")
        access = getattr(mvs, "MV_ACCESS_Exclusive", 1)
        ret = self.cam.MV_CC_OpenDevice(access, 0)
        if ret != 0:
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:x}")
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:x}")

    def _configure(self, width: int = 0, height: int = 0, fps: float = 0.0) -> None:
        mvs = self.mvs
        cam = self.cam
        try:
            cam.MV_CC_SetEnumValue("TriggerMode", getattr(mvs, "MV_TRIGGER_MODE_OFF", 0))
        except Exception:
            pass
        if width > 0:
            try:
                cam.MV_CC_SetIntValue("Width", int(width))
            except Exception:
                print("[HIK][WARN] 设置 Width 失败，忽略")
        if height > 0:
            try:
                cam.MV_CC_SetIntValue("Height", int(height))
            except Exception:
                print("[HIK][WARN] 设置 Height 失败，忽略")
        if fps > 0:
            try:
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
            except Exception:
                print("[HIK][WARN] 设置 AcquisitionFrameRate 失败，忽略")

    def _convert_to_bgr(self, frame_out: Any) -> np.ndarray:
        """Return an OpenCV/TensorRT-ready BGR image.

        BGR8 fast path:
        - If the Hikrobot camera actually outputs PixelType_Gvsp_BGR8_Packed,
          the SDK buffer is already HxWx3 BGR.
        - In that case this function only wraps + copies the SDK buffer and
          intentionally skips MV_CC_ConvertPixelType and cv2.cvtColor.
        - The copy is kept because MV_CC_FreeImageBuffer() releases/reuses the
          SDK buffer immediately after read(), while later YOLO/SHM/UE code may
          still hold the numpy image.

        Fallback:
        - Non-BGR8 camera output keeps the old SDK conversion path, so sync,
          bullet trajectory detection, event trigger, JSONL, SHM and UE output
          logic are not changed by this patch.
        """
        mvs = self.mvs
        info = frame_out.stFrameInfo
        w = int(info.nWidth)
        h = int(info.nHeight)
        src_len = int(info.nFrameLen)
        src_pixel_type = int(info.enPixelType)
        expected_bgr_len = w * h * 3

        raw = np.ctypeslib.as_array((ctypes.c_ubyte * src_len).from_address(frame_out.pBufAddr))

        bgr_pixel = getattr(mvs, "PixelType_Gvsp_BGR8_Packed", None)
        if bgr_pixel is not None and src_pixel_type == int(bgr_pixel):
            if src_len < expected_bgr_len:
                raise RuntimeError(
                    f"BGR8 frame length too small: len={src_len}, expected={expected_bgr_len}, "
                    f"pixel_type={src_pixel_type}"
                )
            if not getattr(self, "_bgr8_fast_path_printed", False):
                self._bgr8_fast_path_printed = True
                cam_tag = getattr(self, "serial", "") or getattr(getattr(self, "cfg", None), "camera_id", "") or "unknown"
                print(
                    f"[HIK][BGR8_FAST][{cam_tag}] direct BGR8_Packed path enabled: "
                    f"{w}x{h}, len={src_len}, pixel_type={src_pixel_type}; "
                    f"skip MV_CC_ConvertPixelType/cvtColor",
                    flush=True,
                )
            return raw[:expected_bgr_len].reshape(h, w, 3).copy()

        rgb_pixel = getattr(mvs, "PixelType_Gvsp_RGB8_Packed", None)
        if rgb_pixel is not None and src_pixel_type == int(rgb_pixel) and src_len >= expected_bgr_len:
            if not getattr(self, "_rgb8_path_printed", False):
                self._rgb8_path_printed = True
                cam_tag = getattr(self, "serial", "") or getattr(getattr(self, "cfg", None), "camera_id", "") or "unknown"
                print(
                    f"[HIK][RGB8_WARN][{cam_tag}] camera returned RGB8_Packed, not BGR8_Packed: "
                    f"{w}x{h}, len={src_len}, pixel_type={src_pixel_type}; "
                    f"using cvtColor RGB->BGR fallback",
                    flush=True,
                )
            img = raw[:expected_bgr_len].reshape(h, w, 3).copy()
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dst_pixel = bgr_pixel
        need_rgb_to_bgr = False
        if dst_pixel is None:
            dst_pixel = getattr(mvs, "PixelType_Gvsp_RGB8_Packed")
            need_rgb_to_bgr = True

        dst_size = expected_bgr_len
        dst_buf = (ctypes.c_ubyte * dst_size)()
        param = mvs.MV_CC_PIXEL_CONVERT_PARAM()
        ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.pSrcData = frame_out.pBufAddr
        param.nSrcDataLen = src_len
        param.enSrcPixelType = src_pixel_type
        param.enDstPixelType = dst_pixel
        param.pDstBuffer = dst_buf
        param.nDstBufferSize = dst_size

        if not getattr(self, "_sdk_convert_path_printed", False):
            self._sdk_convert_path_printed = True
            cam_tag = getattr(self, "serial", "") or getattr(getattr(self, "cfg", None), "camera_id", "") or "unknown"
            print(
                f"[HIK][SDK_CONVERT][{cam_tag}] source pixel_type={src_pixel_type}, len={src_len}; "
                f"using MV_CC_ConvertPixelType -> BGR8 fallback",
                flush=True,
            )
        ret = self.cam.MV_CC_ConvertPixelType(param)
        if ret != 0:
            if src_len == w * h:
                gray = raw.reshape(h, w).copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if src_len >= expected_bgr_len:
                return raw[:expected_bgr_len].reshape(h, w, 3).copy()
            raise RuntimeError(f"MV_CC_ConvertPixelType failed: 0x{ret:x}, pixel_type={src_pixel_type}, len={src_len}")
        img = np.ctypeslib.as_array(dst_buf).reshape(h, w, 3).copy()
        if need_rgb_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        mvs = self.mvs
        frame_out = mvs.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
        ret = self.cam.MV_CC_GetImageBuffer(frame_out, self.timeout_ms)
        if ret != 0:
            return False, None, now_us()
        try:
            frame = self._convert_to_bgr(frame_out)
            return True, frame, now_us()
        finally:
            try:
                self.cam.MV_CC_FreeImageBuffer(frame_out)
            except Exception:
                pass

    def release(self) -> None:
        if self.cam is None:
            return
        for fn in ["MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"]:
            try:
                getattr(self.cam, fn)()
            except Exception:
                pass


def create_source(args: argparse.Namespace) -> BaseFrameSource:
    if args.source == "video":
        if not args.video:
            raise ValueError("source=video 时必须提供 video.path")
        return VideoFileSource(args.video, backend=args.video_backend)
    if args.source == "hik":
        return HikMvsSource(args.serial, args.mvs_sdk_path, args.hik_timeout_ms, args.capture_width, args.capture_height, args.capture_fps)
    raise ValueError(f"未知 source: {args.source}")


def draw_local_region(vis: np.ndarray, projector: Optional[LocalMapProjector], args: argparse.Namespace) -> None:
    if projector is None or not projector.enabled or not bool(args.draw_local_region):
        return
    pts_all = np.asarray(projector.image_points, dtype=np.int32)
    if pts_all.shape[0] >= 1:
        # 标定点可能有 8~9 个，不一定前 4 个就是边界，所以这里只画点和序号，避免画出误导性闭合框。
        for i, (x, y) in enumerate(pts_all.reshape(-1, 2).tolist()):
            cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)
            cv2.putText(vis, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _display_global_id_for_person(p: PersonInstance) -> Optional[int]:
    gid = getattr(p, "global_id_from_A", None)
    try:
        if gid is not None and int(gid) > 0:
            return int(gid)
    except Exception:
        pass
    return None


def _display_color_for_person(p: PersonInstance, args: argparse.Namespace) -> Tuple[int, int, int]:
    gid = _display_global_id_for_person(p)
    if gid is not None:
        return make_color(gid - 1)
    if bool(getattr(args, "a_anchor_reid_enable", False)):
        return (80, 80, 255)  # unknown 显示红色，提醒没有匹配到 A 库
    return make_color(int(p.instance_id) - 1)


def _display_label_for_person(p: PersonInstance, args: argparse.Namespace) -> str:
    gid = _display_global_id_for_person(p)
    if bool(getattr(args, "a_anchor_reid_enable", False)):
        if gid is not None:
            base = f"P{gid}"
        else:
            mode = str(getattr(args, "a_anchor_reid_unknown_label_mode", "unknown") or "unknown").lower()
            if mode == "local":
                base = f"U{int(getattr(p, 'local_stable_id', None) or p.instance_id)}"
            elif str(getattr(p, "reid_source_anchor", "")) == "A_GROUP_WAIT":
                base = "WAIT"
            else:
                base = "UNK"
        if bool(getattr(args, "a_anchor_reid_show_score", False)) and getattr(p, "reid_score", 0.0) > 0:
            base += f" {float(p.reid_score):.2f}"
    else:
        base = f"P{int(p.instance_id)}"
    if bool(args.display_local_xy) and p.local_xy is not None:
        base += f" F({p.local_xy[0]:.2f},{p.local_xy[1]:.2f})"
    return base


def draw_instances(frame_bgr: np.ndarray, persons: List[PersonInstance], args: argparse.Namespace, fps_text: str = "") -> np.ndarray:
    """按参考图风格绘制：人体 mask 填充 + 明显轮廓 + 大号 P1/P2/P3 标签。

    默认不画矩形 bbox；如果没有 segmentation mask，才会看到 bbox 或 NO MASK 提醒。
    """
    vis = frame_bgr.copy()
    if persons and bool(args.draw_mask):
        overlay = vis.copy()
        for p in persons:
            color = _display_color_for_person(p, args)
            if p.mask is not None:
                overlay[p.mask > 0] = color
        vis = cv2.addWeighted(overlay, float(args.mask_alpha), vis, 1.0 - float(args.mask_alpha), 0)

    for p in persons:
        color = _display_color_for_person(p, args)
        if bool(args.draw_contour) and p.contours:
            # 三层轮廓：黑色外边保证复杂背景上也能看清，彩色主体接近参考效果，白色细线增强边缘。
            thick = max(1, int(args.contour_thickness))
            if bool(getattr(args, "draw_contour_shadow", True)):
                cv2.drawContours(vis, p.contours, -1, (0, 0, 0), thick + 3, cv2.LINE_AA)
            cv2.drawContours(vis, p.contours, -1, color, thick, cv2.LINE_AA)
            if bool(getattr(args, "draw_contour_inner_white", True)) and thick >= 3:
                cv2.drawContours(vis, p.contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        if bool(args.draw_box):
            x1, y1, x2, y2 = p.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, max(1, int(getattr(args, "bbox_thickness", 2))), cv2.LINE_AA)
        if bool(args.draw_label):
            try:
                x, y, _, _ = cv2.boundingRect(np.vstack(p.contours)) if p.contours else (*p.bbox_xyxy[:2], 1, 1)
            except Exception:
                x, y = p.bbox_xyxy[0], p.bbox_xyxy[1]
            label = _display_label_for_person(p, args)
            font_scale = float(getattr(args, "label_font_scale", 0.92))
            text_thick = max(1, int(getattr(args, "label_thickness", 2)))
            tx = int(max(4, min(x, vis.shape[1] - 20)))
            ty = int(max(24, y - 8))
            if bool(getattr(args, "label_black_shadow", True)):
                cv2.putText(vis, label, (tx + 2, ty + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thick + 3, cv2.LINE_AA)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thick, cv2.LINE_AA)
        if bool(args.draw_foot_point) and p.foot_pixel is not None:
            cv2.circle(vis, (int(p.foot_pixel[0]), int(p.foot_pixel[1])), max(3, int(getattr(args, "foot_point_radius", 4))), color, -1, cv2.LINE_AA)

    if bool(getattr(args, "draw_no_mask_warning", True)) and persons:
        no_mask_count = sum(1 for p in persons if p.mask is None or int(np.count_nonzero(p.mask)) <= 0)
        if no_mask_count > 0:
            cv2.putText(vis, f"NO SEG MASK: {no_mask_count}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3, cv2.LINE_AA)
    if fps_text:
        cv2.putText(vis, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return vis

def resize_for_display(vis: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if vis is None:
        return vis
    h, w = vis.shape[:2]
    if h <= 0 or w <= 0:
        return vis
    scale = float(args.display_scale or 1.0)
    if bool(args.display_fit_screen):
        if int(args.display_max_width) > 0:
            scale = min(scale, float(args.display_max_width) / float(w))
        if int(args.display_max_height) > 0:
            scale = min(scale, float(args.display_max_height) / float(h))
    scale = min(scale, 1.0)
    if scale < 0.999:
        return cv2.resize(vis, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return vis


def json_record(frame_id: int, ts_us: int, frame_shape: Tuple[int, int, int], persons: List[PersonInstance], source_name: str) -> Dict[str, Any]:
    h, w = frame_shape[:2]
    return {
        "source": source_name,
        "frame_id": int(frame_id),
        "timestamp_us": int(ts_us),
        "frame_w": int(w),
        "frame_h": int(h),
        "persons": [
            {
                "stable_id": int(p.instance_id),
                "local_stable_id": int(getattr(p, "local_stable_id", None) or p.instance_id),
                "global_id_from_A": int(p.global_id_from_A) if getattr(p, "global_id_from_A", None) is not None else None,
                "reid_score": round(float(getattr(p, "reid_score", 0.0)), 4),
                "reid_margin": round(float(getattr(p, "reid_margin", 0.0)), 4),
                "reid_source_anchor": str(getattr(p, "reid_source_anchor", "")),
                "confidence": round(float(p.confidence), 4),
                "bbox_xyxy": [int(v) for v in p.bbox_xyxy],
                "mask_area": int(p.area),
                "foot_pixel": [int(p.foot_pixel[0]), int(p.foot_pixel[1])] if p.foot_pixel is not None else None,
                "local_xy": [round(float(p.local_xy[0]), 3), round(float(p.local_xy[1]), 3)] if p.local_xy is not None else None,
                "visibility": str(p.visibility),
                "mask_polygons": p.polygons,
            }
            for p in persons
        ],
    }


def create_video_writer(path: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    if not path:
        return None
    ensure_dir_for_file(path)
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*( "mp4v" if ext in [".mp4", ".m4v"] else "XVID"))
    writer = cv2.VideoWriter(path, fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {path}")
    return writer



def _resolve_yolo_model_path(model_path: Any, config_path: str = "") -> Tuple[str, bool, str]:
    """Resolve a YOLO model path for logging without accidentally changing model semantics.

    If the user configured an existing local file, return its absolute path so Ultralytics
    loads exactly that file.  If it is a model alias such as "yolo11n-seg.pt" and no local
    file is found, keep the original string and print the searched absolute path.
    """
    raw = str(model_path or "").strip()
    if not raw:
        return raw, False, ""
    expanded = os.path.expanduser(os.path.expandvars(raw))
    candidates = []
    if os.path.isabs(expanded):
        candidates.append(expanded)
    else:
        candidates.append(os.path.abspath(expanded))
        if config_path:
            cfg_dir = os.path.dirname(os.path.abspath(os.path.expanduser(os.path.expandvars(str(config_path)))))
            candidates.append(os.path.abspath(os.path.join(cfg_dir, expanded)))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(script_dir, expanded)))
        candidates.append(os.path.abspath(os.path.join(script_dir, "..", expanded)))
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
            if os.path.exists(c):
                return c, True, c
    display_abs = seen[0] if seen else os.path.abspath(expanded)
    return raw, False, display_abs

def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    source = create_source(args)
    print(f"[source] {source.source_name}")
    model_arg, model_exists, model_abs = _resolve_yolo_model_path(args.model, getattr(args, "config", ""))
    print(f"[model] loading {args.model}")
    print(f"[model] resolved_path={model_abs} exists={model_exists}")
    if model_exists:
        args.model = model_arg
    model = YOLO(args.model)
    stable_manager = StrictStableIDManager(args)
    local_projector = LocalMapProjector(args)
    json_fp = None
    if args.save_json:
        ensure_dir_for_file(args.save_json)
        json_fp = open(args.save_json, "a", encoding="utf-8")
        print(f"[json] save to {args.save_json}")
    writer = None
    frame_id = 0
    fps_t0 = time.time()
    fps_count = 0
    fps_value = 0.0
    last_infer_ms = 0.0
    last_vis = None
    try:
        while True:
            ok, frame, ts_us = source.read()
            if not ok or frame is None:
                if args.source == "video":
                    print("[done] video end")
                    break
                time.sleep(0.002)
                continue
            frame_id += 1
            if args.resize_width > 0 and args.resize_height > 0:
                frame = cv2.resize(frame, (int(args.resize_width), int(args.resize_height)), interpolation=cv2.INTER_AREA)
            do_infer = (frame_id % int(args.process_every_n) == 0)
            persons: List[PersonInstance] = []
            if do_infer:
                persons, last_infer_ms = infer_persons(model, frame, args)
                assign_local_positions(persons, local_projector, args)
                persons = stable_manager.update(persons, frame, ts_us, frame_id)
            fps_count += 1
            if time.time() - fps_t0 >= 1.0:
                fps_value = fps_count / max(1e-6, time.time() - fps_t0)
                fps_t0 = time.time()
                fps_count = 0
            fps_text = f"FPS={fps_value:.1f} infer={last_infer_ms:.1f}ms persons={len(persons)}"
            vis = draw_instances(frame, persons, args, fps_text=fps_text)
            draw_local_region(vis, local_projector, args)
            last_vis = vis
            if json_fp is not None and do_infer:
                json_fp.write(json.dumps(json_record(frame_id, ts_us, frame.shape, persons, source.source_name), ensure_ascii=False) + "\n")
                if bool(args.flush_json):
                    json_fp.flush()
            if writer is None and args.save_video:
                h, w = vis.shape[:2]
                writer = create_video_writer(args.save_video, args.output_fps, (w, h))
                print(f"[video] save to {args.save_video}")
            if writer is not None:
                writer.write(vis)
            if bool(args.show):
                display_vis = resize_for_display(vis, args)
                try:
                    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(args.window_name, display_vis)
                    cv2.resizeWindow(args.window_name, display_vis.shape[1], display_vis.shape[0])
                except Exception:
                    cv2.imshow(args.window_name, display_vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and last_vis is not None:
                    snap = args.snapshot_path or f"person_seg_snapshot_{int(time.time())}.png"
                    cv2.imwrite(snap, last_vis)
                    print(f"[snapshot] saved {snap}")
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if json_fp is not None:
            json_fp.close()
        cv2.destroyAllWindows()


def _norm_key(k: str) -> str:
    return str(k).strip().lstrip("-").replace("-", "_")


def _flatten_config_section(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    sections = {"input", "hik", "video", "model", "mask", "merge", "stable_id", "a_anchor_reid", "local_map", "global_person_fusion", "output", "display", "runtime"}
    out: Dict[str, Any] = {}
    for k, v in data.items():
        nk = _norm_key(k)
        if nk in sections or nk in {"profiles", "common", "defaults", "active_profile", "notes"}:
            continue
        out[nk] = v
    for sec in ["input", "hik", "video", "model", "mask", "merge", "stable_id", "a_anchor_reid", "local_map", "global_person_fusion", "output", "display", "runtime"]:
        obj = data.get(sec)
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            nk = _norm_key(k)
            if sec == "video" and nk in {"path", "file", "input", "input_video"}:
                nk = "video"
            elif sec == "video" and nk in {"backend", "reader", "video_backend"}:
                nk = "video_backend"
            elif sec == "stable_id":
                nk = nk if nk.startswith("stable_id_") else "stable_id_" + nk
            elif sec == "a_anchor_reid":
                nk = nk if nk.startswith("a_anchor_reid_") else "a_anchor_reid_" + nk
            elif sec == "local_map":
                lm_map = {
                    "enable": "local_map_enable", "enabled": "local_map_enable", "calib_path": "local_calib_path",
                    "calibration_path": "local_calib_path", "image_points": "local_image_points",
                    "draw_region": "draw_local_region",
                }
                nk = lm_map.get(nk, nk)
            elif sec == "global_person_fusion":
                nk = nk if nk.startswith("global_person_fusion_") else "global_person_fusion_" + nk
            out[nk] = v
    return out


def _apply_config_dict(args: argparse.Namespace, params: Dict[str, Any], where: str) -> None:
    valid = set(vars(args).keys())
    for raw_key, val in params.items():
        key = _norm_key(raw_key)
        if key not in valid:
            print(f"[config][WARN] unknown key ignored: {where}.{raw_key} -> {key}")
            continue
        setattr(args, key, val)


def _default_config_path() -> str:
    return os.path.splitext(os.path.abspath(__file__))[0] + ".json"


def _load_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    cfg_path = str(args.config or "").strip() or _default_config_path()
    args.config = cfg_path
    if not os.path.exists(cfg_path):
        print(f"[config][WARN] 配置文件不存在，使用默认参数: {cfg_path}")
        return args
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件顶层必须是 JSON object")
    print(f"[config] loaded: {cfg_path}")
    common: Dict[str, Any] = {}
    if isinstance(cfg.get("common"), dict):
        common.update(_flatten_config_section(cfg["common"]))
    common.update(_flatten_config_section(cfg))
    _apply_config_dict(args, common, "common")
    profiles = cfg.get("profiles", {}) or {}
    active = str(cfg.get("active_profile", "") or "").strip()
    if active:
        if not isinstance(profiles, dict) or active not in profiles:
            raise ValueError(f"active_profile={active}，但 profiles 里没有这个配置")
        _apply_config_dict(args, _flatten_config_section(profiles[active]), f"profiles.{active}")
        print(f"[config] active_profile: {active}")
    # normalize
    args.process_every_n = max(1, int(args.process_every_n))
    args.imgsz = int(args.imgsz)
    args.conf = float(args.conf)
    args.iou = float(args.iou)
    args.half = _as_bool(getattr(args, "half", True), True)
    args.retina_masks = _as_bool(getattr(args, "retina_masks", True), True)
    args.max_det = max(1, int(getattr(args, "max_det", 80) or 80))
    args.max_persons = int(args.max_persons)
    args.mask_alpha = float(args.mask_alpha)
    args.video_backend = str(args.video_backend or "auto").lower()
    if args.video_backend not in {"auto", "opencv", "ffmpeg"}:
        args.video_backend = "auto"
    print(f"[config] source={args.source}, model={args.model}, imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, half={args.half}, retina_masks={args.retina_masks}, max_det={args.max_det}, stable_id={args.stable_id_enable}")
    print(f"[config] video={args.video}, video_backend={args.video_backend}, serial={args.serial}")
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单相机 YOLO-seg 人体分割 + 严格稳定 ID")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--source", choices=["hik", "video"], default="hik")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--video-backend", choices=["auto", "opencv", "ffmpeg"], default="auto")
    p.add_argument("--serial", type=str, default="")
    p.add_argument("--mvs-sdk-path", type=str, default="/opt/MVS/Samples/64/Python/MvImport")
    p.add_argument("--hik-timeout-ms", type=int, default=1000)
    p.add_argument("--capture-width", type=int, default=0)
    p.add_argument("--capture-height", type=int, default=0)
    p.add_argument("--capture-fps", type=float, default=0.0)
    p.add_argument("--resize-width", type=int, default=0)
    p.add_argument("--resize-height", type=int, default=0)
    p.add_argument("--model", type=str, default="yolo11n-seg.pt")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--retina-masks", action="store_true", default=True)
    p.add_argument("--max-det", type=int, default=80)
    p.add_argument("--person-class-id", type=int, default=0)
    p.add_argument("--mask-thresh", type=float, default=0.45)
    p.add_argument("--process-every-n", type=int, default=1)
    p.add_argument("--min-mask-area", type=int, default=120)
    p.add_argument("--min-contour-area", type=float, default=60.0)
    p.add_argument("--max-persons", type=int, default=12)
    p.add_argument("--mask-open-iters", type=int, default=0)
    p.add_argument("--mask-open-kernel", type=int, default=3)
    p.add_argument("--mask-close-iters", type=int, default=1)
    p.add_argument("--mask-close-kernel", type=int, default=5)
    p.add_argument("--mask-erode-px", type=int, default=0)
    p.add_argument("--mask-dilate-px", type=int, default=2)
    p.add_argument("--poly-epsilon-ratio", type=float, default=0.003)
    p.add_argument("--mask-split-components", action="store_true", default=False)
    p.add_argument("--mask-split-min-area", type=int, default=160)
    p.add_argument("--mask-split-min-area-ratio", type=float, default=0.08)
    p.add_argument("--mask-split-erode-px", type=int, default=0)
    p.add_argument("--mask-split-restore-dilate-px", type=int, default=0)
    p.add_argument("--mask-split-max-components", type=int, default=4)
    p.add_argument("--mask-split-min-sum-area-ratio", type=float, default=0.45)
    # merge
    p.add_argument("--merge-same-person", action="store_true", default=True)
    p.add_argument("--merge-iou-thresh", type=float, default=0.03)
    p.add_argument("--merge-contain-thresh", type=float, default=0.35)
    p.add_argument("--merge-center-dist-px", type=float, default=95.0)
    p.add_argument("--merge-gap-px", type=int, default=35)
    p.add_argument("--merge-mask-dilate-px", type=int, default=10)
    p.add_argument("--merge-small-area-ratio", type=float, default=0.75)
    p.add_argument("--merge-min-x-overlap-ratio", type=float, default=0.08)
    p.add_argument("--merge-min-y-overlap-ratio", type=float, default=0.08)
    p.add_argument("--merge-close-kernel", type=int, default=5)
    p.add_argument("--merge-close-iters", type=int, default=1)
    p.add_argument("--merge-smart-fragment-enable", action="store_true", default=True)
    p.add_argument("--merge-fragment-area-ratio", type=float, default=0.58)
    p.add_argument("--merge-full-pair-area-ratio", type=float, default=0.62)
    p.add_argument("--merge-full-pair-height-ratio", type=float, default=0.68)
    p.add_argument("--merge-full-pair-min-area-ratio", type=float, default=0.35)
    p.add_argument("--merge-strong-iou-thresh", type=float, default=0.18)
    p.add_argument("--merge-strong-contain-thresh", type=float, default=0.62)
    p.add_argument("--merge-fragment-center-dist-px", type=float, default=105.0)
    # strict stable id
    p.add_argument("--stable-id-enable", action="store_true", default=True)
    p.add_argument("--stable-id-start-id", type=int, default=1)
    p.add_argument("--stable-id-match-score-thresh", type=float, default=0.34)
    p.add_argument("--stable-id-match-iou-thresh", type=float, default=0.025)
    p.add_argument("--stable-id-match-contain-thresh", type=float, default=0.22)
    p.add_argument("--stable-id-match-center-dist-px", type=float, default=240.0)
    p.add_argument("--stable-id-match-local-dist", type=float, default=26.0)
    p.add_argument("--stable-id-match-size-ratio", type=float, default=8.0)
    p.add_argument("--stable-id-enable-color-hist", action="store_true", default=True)
    p.add_argument("--stable-id-appearance-match-thresh", type=float, default=0.50)
    p.add_argument("--stable-id-appearance-ema-alpha", type=float, default=0.90)
    p.add_argument("--stable-id-position-weight", type=float, default=0.25)
    p.add_argument("--stable-id-local-weight", type=float, default=0.42)
    p.add_argument("--stable-id-overlap-weight", type=float, default=0.18)
    p.add_argument("--stable-id-appearance-weight", type=float, default=0.10)
    p.add_argument("--stable-id-size-weight", type=float, default=0.03)
    p.add_argument("--stable-id-recency-weight", type=float, default=0.02)
    p.add_argument("--stable-id-use-local-xy", action="store_true", default=True)
    p.add_argument("--stable-id-inactive-memory-ms", type=float, default=600000.0)
    p.add_argument("--stable-id-keep-lost-forever", action="store_true", default=True)
    p.add_argument("--stable-id-reuse-memory-ms", type=float, default=20000.0)
    p.add_argument("--stable-id-active-max-miss-frames", type=int, default=12)
    p.add_argument("--stable-id-active-grace-ms", type=float, default=1200.0)
    p.add_argument("--stable-id-active-match-score-thresh", type=float, default=0.14)
    p.add_argument("--stable-id-active-center-dist-px", type=float, default=520.0)
    p.add_argument("--stable-id-active-local-dist", type=float, default=5.0)
    p.add_argument("--stable-id-active-size-ratio", type=float, default=12.0)
    p.add_argument("--stable-id-active-partial-boost", type=float, default=1.35)
    p.add_argument("--stable-id-active-position-weight", type=float, default=0.42)
    p.add_argument("--stable-id-active-local-weight", type=float, default=0.32)
    p.add_argument("--stable-id-active-overlap-weight", type=float, default=0.18)
    p.add_argument("--stable-id-active-appearance-weight", type=float, default=0.04)
    p.add_argument("--stable-id-active-size-weight", type=float, default=0.02)
    p.add_argument("--stable-id-active-recency-weight", type=float, default=0.02)
    p.add_argument("--stable-id-bridge-max-ms", type=float, default=1500.0)
    p.add_argument("--stable-id-bridge-match-score-thresh", type=float, default=0.12)
    # 原地遮挡/原地漏检保持：玩家在同一位置被障碍物挡住后重新出现，优先恢复原 ID。
    p.add_argument("--stable-id-hold-enable", action="store_true", default=True)
    p.add_argument("--stable-id-hold-max-ms", type=float, default=8000.0)
    p.add_argument("--stable-id-hold-match-score-thresh", type=float, default=0.18)
    p.add_argument("--stable-id-hold-local-dist", type=float, default=3.2)
    p.add_argument("--stable-id-hold-center-dist-px", type=float, default=360.0)
    p.add_argument("--stable-id-hold-tight-local-dist", type=float, default=1.4)
    p.add_argument("--stable-id-hold-tight-center-dist-px", type=float, default=150.0)
    p.add_argument("--stable-id-hold-appearance-thresh", type=float, default=0.30)
    p.add_argument("--stable-id-hold-noapp-max-ms", type=float, default=2800.0)
    p.add_argument("--stable-id-hold-size-ratio", type=float, default=14.0)
    # v5：多人同时遮挡/障碍物后交叉的保护参数。
    p.add_argument("--stable-id-hold-long-ms", type=float, default=5000.0)
    p.add_argument("--stable-id-hold-long-appearance-thresh", type=float, default=0.42)
    p.add_argument("--stable-id-hold-cross-enable", action="store_true", default=True)
    p.add_argument("--stable-id-hold-cross-min-lost-ms", type=float, default=650.0)
    p.add_argument("--stable-id-hold-cross-local-dist", type=float, default=7.5)
    p.add_argument("--stable-id-hold-cross-center-dist-px", type=float, default=900.0)
    p.add_argument("--stable-id-hold-cross-appearance-thresh", type=float, default=0.58)
    p.add_argument("--stable-id-hold-ambiguous-guard-enable", action="store_true", default=True)
    p.add_argument("--stable-id-hold-ambiguous-min-lost-ms", type=float, default=650.0)
    p.add_argument("--stable-id-hold-min-score-margin", type=float, default=0.055)
    p.add_argument("--stable-id-hold-ambiguous-app-thresh", type=float, default=0.50)
    p.add_argument("--stable-id-hold-ambiguous-app-margin", type=float, default=0.08)
    p.add_argument("--stable-id-lost-reid-enable", action="store_true", default=True)
    p.add_argument("--stable-id-lost-reid-max-ms", type=float, default=5000.0)
    p.add_argument("--stable-id-lost-reid-local-dist", type=float, default=1.35)
    p.add_argument("--stable-id-lost-reid-center-dist-px", type=float, default=230.0)
    p.add_argument("--stable-id-lost-reid-appearance-thresh", type=float, default=0.70)
    p.add_argument("--stable-id-lost-reid-require-appearance", action="store_true", default=True)
    p.add_argument("--stable-id-lost-track-score-penalty", type=float, default=0.12)
    p.add_argument("--stable-id-suppress-duplicate-new", action="store_true", default=True)
    p.add_argument("--stable-id-duplicate-center-dist-px", type=float, default=150.0)
    p.add_argument("--stable-id-duplicate-local-dist", type=float, default=1.6)
    p.add_argument("--stable-id-duplicate-contain-thresh", type=float, default=0.18)
    p.add_argument("--stable-id-duplicate-area-ratio", type=float, default=0.70)
    p.add_argument("--stable-id-new-track-min-area", type=int, default=220)
    p.add_argument("--stable-id-motion-predict-enable", action="store_true", default=True)
    p.add_argument("--stable-id-motion-max-predict-frames", type=float, default=30.0)
    p.add_argument("--stable-id-velocity-ema-alpha", type=float, default=0.60)
    # 当前相机画面边缘判断。注意：真实场地边界不是当前相机可见边界，
    # 所以 edge_related 必须用 bbox 是否靠近图像边缘来判断。
    p.add_argument("--stable-id-use-image-edge", action="store_true", default=True)
    p.add_argument("--stable-id-image-edge-margin-px", type=float, default=70.0)
    p.add_argument("--stable-id-edge-gate-boost", type=float, default=1.7)
    p.add_argument("--stable-id-partial-gate-boost", type=float, default=1.4)
    p.add_argument("--stable-id-strict-reappear-enable", action="store_true", default=True)
    p.add_argument("--stable-id-strict-reappear-after-ms", type=float, default=700.0)
    p.add_argument("--stable-id-strict-reappear-center-dist-px", type=float, default=260.0)
    p.add_argument("--stable-id-strict-reappear-local-dist", type=float, default=28.0)
    p.add_argument("--stable-id-edge-reappear-center-dist-px", type=float, default=620.0)
    p.add_argument("--stable-id-edge-reappear-local-dist", type=float, default=58.0)
    p.add_argument("--stable-id-strict-reappear-appearance-thresh", type=float, default=0.68)
    p.add_argument("--stable-id-new-person-center-jump-px", type=float, default=520.0)
    p.add_argument("--stable-id-new-person-local-jump", type=float, default=45.0)
    # 障碍物遮挡重识别：P4 完全被障碍物挡住后，从同一障碍物另一侧出现时允许恢复 P4。
    p.add_argument("--stable-id-occlusion-enable", action="store_true", default=True)
    p.add_argument("--stable-id-occlusion-zones", type=list, default=[])
    p.add_argument("--stable-id-occlusion-margin", type=float, default=8.0)
    p.add_argument("--stable-id-occlusion-memory-ms", type=float, default=4500.0)
    p.add_argument("--stable-id-occlusion-min-lost-ms", type=float, default=250.0)
    p.add_argument("--stable-id-occlusion-center-dist-px", type=float, default=760.0)
    p.add_argument("--stable-id-occlusion-local-dist", type=float, default=68.0)
    p.add_argument("--stable-id-occlusion-appearance-thresh", type=float, default=0.42)
    p.add_argument("--stable-id-occlusion-score-bonus", type=float, default=0.16)
    # A-only anchor ReID: A 相机建身份库，B/C/D/E 只匹配 A 身份库。
    p.add_argument("--a-anchor-reid-enable", action="store_true", default=False)
    p.add_argument("--a-anchor-reid-anchor-camera", type=str, default="A")
    p.add_argument("--a-anchor-reid-allow-non-anchor-enroll", action="store_true", default=False)
    p.add_argument("--a-anchor-reid-start-id", type=int, default=1)
    p.add_argument("--a-anchor-reid-max-ids", type=int, default=64)
    p.add_argument("--a-anchor-reid-feature-ema-alpha", type=float, default=0.86)
    p.add_argument("--a-anchor-reid-min-feature-area", type=int, default=80)
    p.add_argument("--a-anchor-reid-upper-ratio", type=float, default=0.58)
    p.add_argument("--a-anchor-reid-lower-start-ratio", type=float, default=0.42)
    p.add_argument("--a-anchor-reid-whole-weight", type=float, default=0.45)
    p.add_argument("--a-anchor-reid-upper-weight", type=float, default=0.35)
    p.add_argument("--a-anchor-reid-lower-weight", type=float, default=0.20)
    p.add_argument("--a-anchor-reid-anchor-min-conf", type=float, default=0.35)
    p.add_argument("--a-anchor-reid-anchor-min-area", type=int, default=220)
    p.add_argument("--a-anchor-reid-anchor-match-thresh", type=float, default=0.72)
    p.add_argument("--a-anchor-reid-anchor-min-margin", type=float, default=0.035)
    p.add_argument("--a-anchor-reid-query-min-conf", type=float, default=0.18)
    p.add_argument("--a-anchor-reid-query-min-area", type=int, default=80)
    p.add_argument("--a-anchor-reid-query-match-thresh", type=float, default=0.58)
    p.add_argument("--a-anchor-reid-query-min-margin", type=float, default=0.015)
    p.add_argument("--a-anchor-reid-show-score", action="store_true", default=False)
    p.add_argument("--a-anchor-reid-unknown-label-mode", type=str, default="unknown")
    # A-anchor 外观特征后端：hsv=原轻量颜色直方图；osnet=torchreid OSNet 深度 ReID。
    p.add_argument("--a-anchor-reid-backend", type=str, default="hsv")
    p.add_argument("--a-anchor-reid-osnet-model", type=str, default="osnet_x0_25")
    p.add_argument("--a-anchor-reid-osnet-source-dir", type=str, default="")
    p.add_argument("--a-anchor-reid-osnet-weights", type=str, default="")
    p.add_argument("--a-anchor-reid-osnet-local-only", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-osnet-device", type=str, default="auto")
    p.add_argument("--a-anchor-reid-osnet-input-h", type=int, default=256)
    p.add_argument("--a-anchor-reid-osnet-input-w", type=int, default=128)
    p.add_argument("--a-anchor-reid-osnet-batch-size", type=int, default=16)
    p.add_argument("--a-anchor-reid-osnet-half", action="store_true", default=False)
    p.add_argument("--a-anchor-reid-osnet-pretrained", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-osnet-mask-crop", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-osnet-mask-bg", type=int, default=114)
    p.add_argument("--a-anchor-reid-osnet-crop-expand", type=float, default=0.08)
    p.add_argument("--a-anchor-reid-osnet-required", action="store_true", default=False)
    p.add_argument("--a-anchor-reid-osnet-fallback-to-hsv", action="store_true", default=True)
    # A-anchor ReID 低频/缓存模式：YOLO mask 每帧更新，但外观特征不再每帧计算。
    p.add_argument("--a-anchor-reid-lowfreq-enable", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-feature-roi-enable", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-anchor-update-every-n", type=int, default=5)
    p.add_argument("--a-anchor-reid-confirmed-recheck-every-n", type=int, default=10)
    p.add_argument("--a-anchor-reid-unknown-recheck-every-n", type=int, default=2)
    p.add_argument("--a-anchor-reid-cache-hold-max-ms", type=float, default=30000.0)
    p.add_argument("--a-anchor-reid-keep-confirmed-on-recheck-fail", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-switch-match-thresh", type=float, default=0.78)
    p.add_argument("--a-anchor-reid-switch-min-margin", type=float, default=0.06)
    # A-anchor 查询保护：以 A 相机当前 local_xy 作为辅助约束，并且新 local_id 连续命中后才确认。
    p.add_argument("--a-anchor-reid-use-anchor-local-xy-gate", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-local-xy-soft-dist", type=float, default=2.5)
    p.add_argument("--a-anchor-reid-local-xy-hard-dist", type=float, default=6.0)
    p.add_argument("--a-anchor-reid-local-xy-score-weight", type=float, default=0.22)
    p.add_argument("--a-anchor-reid-local-xy-max-age-frames", type=int, default=3)
    p.add_argument("--a-anchor-reid-query-confirm-hits", type=int, default=2)
    p.add_argument("--a-anchor-reid-anchor-update-self-check", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-anchor-update-min-self-sim", type=float, default=0.50)
    # 开局多人粘连保护：A 相机只有一个大 mask 时先不建库，等分开后再分配 P1/P2/P3。
    p.add_argument("--a-anchor-reid-group-guard-enable", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-group-guard-no-enroll-single", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-group-guard-reset-after-split", action="store_true", default=True)
    p.add_argument("--a-anchor-reid-group-guard-split-window-ms", type=float, default=6000.0)
    p.add_argument("--a-anchor-reid-group-guard-reset-if-bank-size-le", type=int, default=2)
    p.add_argument("--a-anchor-reid-group-guard-single-max-aspect", type=float, default=0.78)
    p.add_argument("--a-anchor-reid-group-guard-min-width-ratio", type=float, default=0.10)
    p.add_argument("--a-anchor-reid-group-guard-single-max-width-ratio", type=float, default=0.18)
    p.add_argument("--a-anchor-reid-group-guard-single-max-area-ratio", type=float, default=0.018)

    # local map
    p.add_argument("--local-map-enable", action="store_true", default=True)
    p.add_argument("--local-calib-path", type=str, default="")
    p.add_argument("--local-image-points", type=list, default=[])
    p.add_argument("--local-points", type=list, default=[[0, 100], [100, 100], [100, 0], [0, 0]])
    p.add_argument("--draw-local-region", action="store_true", default=True)
    p.add_argument("--display-local-xy", action="store_true", default=True)
    p.add_argument("--draw-foot-point", action="store_true", default=True)
    p.add_argument("--foot-band-px", type=int, default=8)
    p.add_argument("--full-body-min-h-px", type=int, default=180)
    p.add_argument("--partial-body-min-h-px", type=int, default=80)
    p.add_argument("--position-anchor-mode", type=str, default="topdown_center")
    p.add_argument("--topdown-anchor-method", type=str, default="mask_distance_center")
    p.add_argument("--topdown-erode-px", type=int, default=2)
    p.add_argument("--camera-center-weight-enable", action="store_true", default=True)
    p.add_argument("--no-camera-center-weight-enable", dest="camera_center_weight_enable", action="store_false")
    p.add_argument("--camera-center-weight-power", type=float, default=2.0)
    p.add_argument("--camera-center-min-weight", type=float, default=0.15)
    # Global top-down observation fusion diagnostics. The independent
    # global_person_id_server still owns long-lived global IDs; this annotates
    # per-camera observations so that downstream fusion can prefer reliable ones.
    p.add_argument("--global-person-fusion-enable", action="store_true", default=True)
    p.add_argument("--no-global-person-fusion-enable", dest="global_person_fusion_enable", action="store_false")
    p.add_argument("--global-person-fusion-cluster-dist-m", type=float, default=0.8)
    p.add_argument("--global-person-fusion-outlier-dist-m", type=float, default=1.5)
    p.add_argument("--global-person-fusion-mode", type=str, default="best_center_weight")
    p.add_argument("--global-person-fusion-display-only-confirmed", action="store_true", default=False)
    # output
    p.add_argument("--show", action="store_true", default=True)
    p.add_argument("--window-name", type=str, default="RGB_YOLO_SEG_UNIQUEID_OCCLUSION")
    p.add_argument("--draw-mask", action="store_true", default=True)
    p.add_argument("--draw-contour", action="store_true", default=True)
    p.add_argument("--draw-box", action="store_true", default=False)
    p.add_argument("--draw-label", action="store_true", default=True)
    p.add_argument("--mask-alpha", type=float, default=0.35)
    p.add_argument("--contour-thickness", type=int, default=2)
    p.add_argument("--display-fit-screen", action="store_true", default=True)
    p.add_argument("--display-max-width", type=int, default=1280)
    p.add_argument("--display-max-height", type=int, default=720)
    p.add_argument("--display-scale", type=float, default=1.0)
    p.add_argument("--save-video", type=str, default="")
    p.add_argument("--output-fps", type=float, default=25.0)
    p.add_argument("--save-json", type=str, default="")
    p.add_argument("--flush-json", action="store_true", default=True)
    p.add_argument("--snapshot-path", type=str, default="")
    args = p.parse_args()
    return _load_config_into_args(args)


# =============================================================================
# Clean direct multi-camera / multi-video runner
# 设计目标：
#   1) 多路 HIK / video 输入；
#   2) HIK：CPU + 海康 MVS SDK 取流，MV_CC_GetImageBuffer；
#   3) HIK：CPU + 海康 MVS SDK 格式转换，MV_CC_ConvertPixelType -> BGR8；
#   4) YOLO：单模型 batch 推理，device="0" 用 GPU；
#   5) 后处理 / ID / 画图：CPU；
#   6) 不做拼接 mosaic，不排列；每个相机一个独立窗口，按比例缩小显示完整画面。
# =============================================================================

import copy
import threading
import traceback
from collections import deque


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_direct_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件顶层必须是 JSON object")
    active = str(cfg.get("active_profile", "") or "").strip()
    if active:
        profiles = cfg.get("profiles", {}) or {}
        if active not in profiles:
            raise ValueError(f"active_profile={active}，但 profiles 中没有该配置")
        merged = copy.deepcopy(cfg)
        _deep_update(merged, profiles[active] or {})
        merged["active_profile"] = active
        return merged
    return cfg


def _make_base_default_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--config", "/tmp/nonexistent_singlecam_config_for_direct_multi.json"]
        return parse_args()
    finally:
        sys.argv = old_argv


def _apply_sections_to_args(args: argparse.Namespace, section_cfg: Dict[str, Any], where: str) -> None:
    if not isinstance(section_cfg, dict):
        return
    flat = _flatten_config_section(section_cfg)
    _apply_config_dict(args, flat, where)


# =============================================================================
# Local sync IPC helpers for event-camera / MVS / human-region linkage
# =============================================================================

def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return bool(default)




def _yolo_retina_masks(args: argparse.Namespace) -> bool:
    return _as_bool(getattr(args, "retina_masks", True), True)


def _yolo_half(args: argparse.Namespace) -> bool:
    return _as_bool(getattr(args, "half", True), True)


def _yolo_max_det(args: argparse.Namespace) -> int:
    try:
        return max(1, int(getattr(args, "max_det", 80) or 80))
    except Exception:
        return 80


def _build_yolo_predict_kwargs(source: Any, args: argparse.Namespace, device: Optional[str] = None) -> Dict[str, Any]:
    """Build Ultralytics predict kwargs from config.

    Previous versions hard-coded retina_masks=True and never forwarded half/max_det,
    so common.model.half/max_det in the JSON had no effect.  Keeping this in one
    helper makes single-camera and multi-camera batch paths use identical settings.
    """
    kwargs: Dict[str, Any] = dict(
        source=source,
        imgsz=int(getattr(args, "imgsz", 960)),
        conf=float(getattr(args, "conf", 0.2)),
        iou=float(getattr(args, "iou", 0.45)),
        classes=[int(getattr(args, "person_class_id", 0))],
        retina_masks=_yolo_retina_masks(args),
        half=_yolo_half(args),
        max_det=_yolo_max_det(args),
        verbose=False,
    )
    dev = str(device if device is not None else getattr(args, "device", "") or "").strip()
    if dev:
        kwargs["device"] = dev
    return kwargs

def _sync_enabled(sync_cfg: Dict[str, Any]) -> bool:
    return isinstance(sync_cfg, dict) and _as_bool(sync_cfg.get("enable", False), False)


def _soft_trigger_enabled(sync_cfg: Dict[str, Any]) -> bool:
    """是否使用中心 40ms 软件触发 MVS。

    开启后，所有 HIK 路由由同一个 SoftTriggerCaptureGroup 打开和触发，
    不再让每个 DirectFrameWorker 自己连续采集，避免 A/D 两台 MVS 各跑各的节拍。
    """
    return _sync_enabled(sync_cfg) and _as_bool(sync_cfg.get("mvs_soft_trigger_enable", False), False)


def _sync_path(sync_cfg: Dict[str, Any], key: str, camera: str, default_template: str) -> str:
    root = str(sync_cfg.get("root", "/home/ysxq/PycharmProjects/Ids_Test_3.9/test607/sync_ipc") or "")
    template = str(sync_cfg.get(key, default_template) or default_template)
    path = template.format(camera=str(camera), cam=str(camera), root=root)
    return os.path.abspath(os.path.expanduser(path))


def _event_serial_for_camera(sync_cfg: Dict[str, Any], camera: str) -> str:
    event_serials = sync_cfg.get("event_serials", {}) if isinstance(sync_cfg, dict) else {}
    if isinstance(event_serials, dict):
        return str(event_serials.get(str(camera), "") or "")
    return ""


def _event_name_for_camera(sync_cfg: Dict[str, Any], camera: str) -> str:
    event_names = sync_cfg.get("event_names", {}) if isinstance(sync_cfg, dict) else {}
    if isinstance(event_names, dict):
        return str(event_names.get(str(camera), f"EVT_{camera}") or f"EVT_{camera}")
    return f"EVT_{camera}"


def _wait_for_event_ready_files(sync_cfg: Dict[str, Any], route_order: List[str], route_args: Dict[str, argparse.Namespace]) -> None:
    if not _sync_enabled(sync_cfg) or not _as_bool(sync_cfg.get("wait_event_ready", False), False):
        return
    timeout_sec = float(sync_cfg.get("ready_timeout_sec", 120.0) or 0.0)
    poll_sec = max(0.02, float(sync_cfg.get("ready_poll_sec", 0.1) or 0.1))
    cameras = [cid for cid in route_order if getattr(route_args[cid], "source", "") == "hik"]
    if not cameras:
        return
    deadline = None if timeout_sec <= 0 else time.time() + timeout_sec
    print(f"[sync] wait event ready flags for cameras: {cameras}", flush=True)
    while True:
        missing = []
        for cid in cameras:
            ready = _sync_path(sync_cfg, "ready_file_template", cid, "{root}/event_ready_{camera}.flag")
            if not os.path.exists(ready):
                missing.append(ready)
        if not missing:
            print("[sync] all event ready flags detected; opening MVS cameras...", flush=True)
            return
        if deadline is not None and time.time() > deadline:
            msg = "\n".join(missing)
            raise TimeoutError(f"wait event ready timeout. missing ready flag(s):\n{msg}")
        time.sleep(poll_sec)


def _open_human_jsonl_writers(sync_cfg: Dict[str, Any], route_order: List[str], route_args: Dict[str, argparse.Namespace]) -> Dict[str, Any]:
    writers: Dict[str, Any] = {}
    if not _sync_enabled(sync_cfg) or not _as_bool(sync_cfg.get("write_human_jsonl", True), True):
        return writers
    for cid in route_order:
        if getattr(route_args[cid], "source", "") != "hik":
            continue
        path = _sync_path(sync_cfg, "human_jsonl_template", cid, "{root}/human_result_{camera}.jsonl")
        ensure_dir_for_file(path)
        mode = "w" if _as_bool(sync_cfg.get("truncate_human_jsonl_on_start", True), True) else "a"
        writers[cid] = open(path, mode, encoding="utf-8")
        print(f"[sync][{cid}] human result jsonl -> {path}", flush=True)
    return writers


def _close_jsonl_writers(writers: Dict[str, Any]) -> None:
    for fp in writers.values():
        try:
            fp.close()
        except Exception:
            pass


class YoloInferPerfWriter:
    """Append one CSV row per YOLO batch.

    This is the quickest way to prove whether human_result coverage is limited by
    YOLO throughput.  It records the exact predict settings that were actually
    passed to model.predict(), plus batch sync ids and per-camera person counts.
    """

    fieldnames = [
        "wall_us",
        "batch_index",
        "batch_size",
        "collect_wait_ms",
        "batch_missing_slots",
        "predict_ms",
        "postprocess_batch_ms",
        "total_batch_ms",
        "predict_fps_total",
        "total_fps_total",
        "model",
        "device",
        "imgsz",
        "half",
        "retina_masks",
        "max_det",
        "cam_ids",
        "sync_indices",
        "mvs_frame_nums",
        "frame_ids_local",
        "person_counts",
    ]

    def __init__(self, sync_cfg: Dict[str, Any]):
        self.sync_cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.enabled = _sync_enabled(self.sync_cfg) and _as_bool(self.sync_cfg.get("write_yolo_infer_perf_csv", True), True)
        self.flush = _as_bool(self.sync_cfg.get("yolo_infer_perf_flush", True), True)
        self.fp = None
        self.writer = None
        if not self.enabled:
            return
        path = _sync_path(self.sync_cfg, "yolo_infer_perf_csv_path", "ALL", "{root}/yolo_infer_perf.csv")
        ensure_dir_for_file(path)
        mode = "w" if _as_bool(self.sync_cfg.get("truncate_yolo_infer_perf_on_start", True), True) else "a"
        self.fp = open(path, mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.fieldnames)
        if mode == "w" or os.path.getsize(path) == 0:
            self.writer.writeheader()
        print(f"[sync] yolo infer perf csv -> {path}", flush=True)

    def write(self,
              batch_index: int,
              common_args: argparse.Namespace,
              device: str,
              metas: List[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace, Optional[Dict[str, Any]]]],
              person_counts: List[int],
              predict_ms: float,
              postprocess_batch_ms: float,
              total_batch_ms: float,
              collect_wait_ms: float = 0.0) -> None:
        if self.writer is None or self.fp is None:
            return
        cam_ids: List[str] = []
        sync_indices: List[str] = []
        mvs_frame_nums: List[str] = []
        fids: List[str] = []
        for meta in metas:
            cam_id, _ts_us, fid, mvs_frame_num, mvs_meta, _frame, _args, _event_item = meta
            cam_ids.append(str(cam_id))
            fids.append(str(int(fid)))
            mvs_frame_nums.append(str(int(mvs_frame_num)))
            sync_indices.append(str((mvs_meta or {}).get("program_sync_index", "")))
        bs = max(1, int(len(metas)))
        row = {
            "wall_us": now_us(),
            "batch_index": int(batch_index),
            "batch_size": int(len(metas)),
            "collect_wait_ms": round(float(collect_wait_ms), 3),
            "batch_missing_slots": int(max(0, int(getattr(common_args, "max_batch", 0) or 0) - int(len(metas)))) if int(getattr(common_args, "max_batch", 0) or 0) > 0 else "",
            "predict_ms": round(float(predict_ms), 3),
            "postprocess_batch_ms": round(float(postprocess_batch_ms), 3),
            "total_batch_ms": round(float(total_batch_ms), 3),
            "predict_fps_total": round(1000.0 * bs / max(1e-6, float(predict_ms)), 3),
            "total_fps_total": round(1000.0 * bs / max(1e-6, float(total_batch_ms)), 3),
            "model": str(getattr(common_args, "model", "") or ""),
            "device": str(device or getattr(common_args, "device", "") or ""),
            "imgsz": int(getattr(common_args, "imgsz", 0) or 0),
            "half": int(_yolo_half(common_args)),
            "retina_masks": int(_yolo_retina_masks(common_args)),
            "max_det": int(_yolo_max_det(common_args)),
            "cam_ids": "|".join(cam_ids),
            "sync_indices": "|".join(sync_indices),
            "mvs_frame_nums": "|".join(mvs_frame_nums),
            "frame_ids_local": "|".join(fids),
            "person_counts": "|".join(str(int(x)) for x in person_counts),
        }
        self.writer.writerow(row)
        if self.flush:
            self.fp.flush()

    def close(self) -> None:
        if self.fp is not None:
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None
            self.writer = None


def _person_to_sync_json(p: PersonInstance, include_polygons: bool = True) -> Dict[str, Any]:
    rec = {
        "person_id": int(p.instance_id),
        "stable_id": int(p.instance_id),
        "local_stable_id": int(getattr(p, "local_stable_id", None) or p.instance_id),
        "global_id_from_A": int(p.global_id_from_A) if getattr(p, "global_id_from_A", None) is not None else None,
        "reid_score": round(float(getattr(p, "reid_score", 0.0)), 4),
        "reid_margin": round(float(getattr(p, "reid_margin", 0.0)), 4),
        "reid_source_anchor": str(getattr(p, "reid_source_anchor", "")),
        "confidence": round(float(p.confidence), 4),
        "bbox_xyxy": [int(v) for v in p.bbox_xyxy],
        "mask_area": int(p.area),
        "foot_pixel": [int(p.foot_pixel[0]), int(p.foot_pixel[1])] if p.foot_pixel is not None else None,
        "local_xy": [round(float(p.local_xy[0]), 3), round(float(p.local_xy[1]), 3)] if p.local_xy is not None else None,
        "local_xy_raw": [round(float(p.local_xy_raw[0]), 3), round(float(p.local_xy_raw[1]), 3)] if getattr(p, "local_xy_raw", None) is not None else None,
        "fused_xy": [round(float(p.fused_xy[0]), 3), round(float(p.fused_xy[1]), 3)] if getattr(p, "fused_xy", None) is not None else None,
        "anchor_pixel_mode": str(getattr(p, "anchor_pixel_mode", "")),
        "anchor_pixel": [int(p.foot_pixel[0]), int(p.foot_pixel[1])] if p.foot_pixel is not None else None,
        "camera_center_weight": round(float(getattr(p, "camera_center_weight", 1.0)), 4),
        "coord_quality": str(getattr(p, "coord_quality", "")),
        "fusion_cluster_id": int(p.fusion_cluster_id) if getattr(p, "fusion_cluster_id", None) is not None else None,
        "fusion_state": str(getattr(p, "fusion_state", "")),
        "fusion_source_cam": str(getattr(p, "fusion_source_cam", "")),
        "fusion_source_cams": list(getattr(p, "fusion_source_cams", None) or []),
        "fusion_center_weight": round(float(getattr(p, "fusion_center_weight", 0.0)), 4),
        "fusion_outlier": bool(getattr(p, "fusion_outlier", False)),
        "visibility": str(p.visibility),
    }
    if include_polygons:
        rec["mask_polygons"] = p.polygons
    return rec


def _build_human_sync_record(cam_id: str,
                             args: argparse.Namespace,
                             sync_cfg: Dict[str, Any],
                             fid: int,
                             mvs_frame_num: int,
                             mvs_meta: Dict[str, Any],
                             ts_us: int,
                             frame_shape: Tuple[int, int, int],
                             persons: List[PersonInstance],
                             infer_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    h, w = frame_shape[:2]
    offset = int(sync_cfg.get("sync_index_offset", -1) or 0) if isinstance(sync_cfg, dict) else -1
    # 连续采集版：sync_index_hint = mvs_frame_num + offset，一般为 mvs_frame_num - 1。
    # 中心软件触发版：优先使用程序统一生成的 program_sync_index，
    # 这样 A/D 两台 MVS 共用同一个 40ms 主节拍。
    program_sync_index = None
    if isinstance(mvs_meta, dict):
        program_sync_index = mvs_meta.get("program_sync_index", None)
    if program_sync_index is not None:
        sync_hint = int(program_sync_index)
    elif mvs_frame_num is None or int(mvs_frame_num) <= 0:
        sync_hint = int(fid) + offset
    else:
        sync_hint = int(mvs_frame_num) + offset
    include_polygons = _as_bool(sync_cfg.get("include_mask_polygons", True), True) if isinstance(sync_cfg, dict) else True
    exposure_us = sync_cfg.get("exposure_us", None) if isinstance(sync_cfg, dict) else None
    try:
        exposure_us = int(exposure_us) if exposure_us is not None else None
    except Exception:
        exposure_us = None
    infer_meta = infer_meta or {}
    rec = {
        "msg_type": "person_regions",
        "camera": str(cam_id),
        "camera_name": str(cam_id),
        "mvs_serial": str(getattr(args, "serial", "") or ""),
        "event_camera": _event_name_for_camera(sync_cfg, cam_id),
        "event_serial": _event_serial_for_camera(sync_cfg, cam_id),
        "frame_id_local": int(fid),
        "mvs_frame_num": int(mvs_frame_num) if mvs_frame_num is not None else int(fid),
        "sync_index_hint": int(sync_hint),
        "sync_index_offset_used": int(offset),
        "capture_wall_us": int(ts_us),
        "record_wall_us": now_us(),
        "exposure_us": exposure_us,
        "frame_w": int(w),
        "frame_h": int(h),
        "mvs_meta": mvs_meta or {},
        "persons": [_person_to_sync_json(p, include_polygons=include_polygons) for p in persons],
        "person_count": int(len(persons)),
        "valid": True,
        "infer_source": str(infer_meta.get("infer_source", "yolo" if infer_meta else "unknown")),
        "yolo_batch_index": infer_meta.get("batch_index", None),
        "yolo_batch_size": infer_meta.get("batch_size", None),
        "yolo_collect_wait_ms": infer_meta.get("collect_wait_ms", None),
        "yolo_predict_ms": infer_meta.get("predict_ms", None),
        "yolo_postprocess_batch_ms": infer_meta.get("postprocess_batch_ms", None),
        "yolo_total_batch_ms": infer_meta.get("total_batch_ms", None),
        "yolo_imgsz": int(getattr(args, "imgsz", 0) or 0),
        "yolo_half": _yolo_half(args),
        "yolo_retina_masks": _yolo_retina_masks(args),
        "yolo_max_det": _yolo_max_det(args),
        "yolo_device": str(getattr(args, "device", "") or ""),
    }
    # Remove None-valued runtime fields to keep JSONL compact while preserving the
    # fields for real YOLO records.
    for _k in ["yolo_batch_index", "yolo_batch_size", "yolo_collect_wait_ms", "yolo_predict_ms", "yolo_postprocess_batch_ms", "yolo_total_batch_ms"]:
        if rec.get(_k, None) is None:
            rec.pop(_k, None)
    return rec


def _append_human_sync_record(writers: Dict[str, Any], cam_id: str, record: Dict[str, Any], flush: bool = True) -> None:
    fp = writers.get(cam_id)
    if fp is None:
        return
    fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    if flush:
        fp.flush()



def _human_carry_enabled(sync_cfg: Dict[str, Any]) -> bool:
    """Whether to synthesize short-gap carried human_result rows.

    Real YOLO-seg may only cover every other MVS frame under load.  For display and
    hit judgement, a short carry is usually better than a hard missing record, but
    every carried row is explicitly marked with infer_source="carry_last" so the
    judge/debug tooling can treat it as weaker evidence than a true YOLO frame.
    """
    if not isinstance(sync_cfg, dict):
        return False
    return _as_bool(sync_cfg.get("human_carry_enable", True), True)


def _human_carry_max_gap_sync(sync_cfg: Dict[str, Any]) -> int:
    if not isinstance(sync_cfg, dict):
        return 0
    try:
        return max(0, int(sync_cfg.get("human_carry_max_gap_sync", 2) or 0))
    except Exception:
        return 2


def _record_sync_index(rec: Dict[str, Any], default: int = -1) -> int:
    try:
        return int(rec.get("sync_index_hint", rec.get("sync_index", default)))
    except Exception:
        meta = rec.get("mvs_meta") if isinstance(rec.get("mvs_meta"), dict) else {}
        try:
            return int(meta.get("program_sync_index", default))
        except Exception:
            return int(default)


def _make_carry_human_record(prev_rec: Dict[str, Any], target_sync: int, sync_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Clone a previous true YOLO human_result into a marked carry_last row.

    This intentionally keeps the previous mask/polygon coordinates.  It is a
    real-time short-term hold, not a new detection.  Consumers must check
    infer_source/human_carry_* fields before making strong decisions.
    """
    rec = copy.deepcopy(prev_rec)
    prev_sync = _record_sync_index(prev_rec, int(target_sync) - 1)
    delta = int(target_sync) - int(prev_sync)
    period_us = int(float(sync_cfg.get("mvs_soft_trigger_period_ms", 40.0) or 40.0) * 1000.0) if isinstance(sync_cfg, dict) else 40000

    rec["sync_index_hint"] = int(target_sync)
    rec["record_wall_us"] = now_us()
    rec["infer_source"] = "carry_last"
    rec["human_carry"] = True
    rec["human_carry_mode"] = "carry_last"
    rec["human_carry_source_sync_index"] = int(prev_sync)
    rec["human_carry_delta_sync"] = int(delta)
    rec["human_carry_max_gap_sync"] = _human_carry_max_gap_sync(sync_cfg)
    rec["human_carry_note"] = "short-gap carried previous YOLO result; weaker than real yolo"

    # Keep frame ids monotonic when possible so per-frame audits line up with MVS.
    try:
        rec["frame_id_local"] = int(prev_rec.get("frame_id_local", 0)) + int(delta)
    except Exception:
        pass
    try:
        rec["mvs_frame_num"] = int(prev_rec.get("mvs_frame_num", 0)) + int(delta)
    except Exception:
        pass
    try:
        rec["capture_wall_us"] = int(prev_rec.get("capture_wall_us", 0)) + int(delta) * int(period_us)
    except Exception:
        pass

    meta = rec.get("mvs_meta") if isinstance(rec.get("mvs_meta"), dict) else {}
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["program_sync_index"] = int(target_sync)
        meta["human_carry_from_sync_index"] = int(prev_sync)
        rec["mvs_meta"] = meta

    # These timing fields belong to the original YOLO batch. Preserve provenance,
    # but don't let audit treat this as a fresh batch timing sample.
    if "yolo_batch_index" in rec:
        rec["carried_from_yolo_batch_index"] = rec.get("yolo_batch_index")
    rec.pop("yolo_collect_wait_ms", None)
    rec.pop("yolo_predict_ms", None)
    rec.pop("yolo_postprocess_batch_ms", None)
    rec.pop("yolo_total_batch_ms", None)
    return rec


def _emit_human_carry_gap_records(
    human_writers: Dict[str, Any],
    cam_id: str,
    prev_real_rec: Optional[Dict[str, Any]],
    current_real_rec: Dict[str, Any],
    sync_cfg: Dict[str, Any],
    flush: bool = True,
) -> int:
    """Write carry_last rows for small missing sync gaps before current_real_rec.

    Returns the number of carry rows written.  Only gaps after the previous true
    YOLO record are filled, and only up to human_carry_max_gap_sync.
    """
    if not human_writers or not _human_carry_enabled(sync_cfg):
        return 0
    if prev_real_rec is None:
        return 0
    prev_sync = _record_sync_index(prev_real_rec, -1)
    curr_sync = _record_sync_index(current_real_rec, -1)
    max_gap = _human_carry_max_gap_sync(sync_cfg)
    if prev_sync < 0 or curr_sync <= prev_sync + 1 or max_gap <= 0:
        return 0
    written = 0
    last_target = min(int(curr_sync), int(prev_sync) + 1 + int(max_gap))
    for target_sync in range(int(prev_sync) + 1, last_target):
        carry = _make_carry_human_record(prev_real_rec, int(target_sync), sync_cfg)
        _append_human_sync_record(human_writers, cam_id, carry, flush=flush)
        written += 1
    return written




class MvsFrameAuditWriter:
    """Write one lightweight CSV row for every captured MVS frame.

    This is intentionally independent from human_result_*.jsonl.  The human JSONL
    is written only when a frame enters YOLO and returns a result, so it cannot by
    itself answer "did every MVS frame have human/event data?".

    The final per-frame bundle table is generated by tools/analyze_frame_bundle_audit.py,
    which joins these MVS rows with:
      - sync_ipc/event_trigger_{camera}.csv
      - sync_ipc/human_result_{camera}.jsonl
    """

    FIELDNAMES = [
        "write_wall_us",
        "camera",
        "frame_id_local",
        "mvs_frame_num",
        "sync_index",
        "capture_wall_us",
        "receiver_wall_us",
        "soft_trigger_tick_wall_us",
        "soft_trigger_done_wall_us",
        "soft_trigger_cmd_wall_us",
        "soft_trigger_cmd_done_wall_us",
        "soft_trigger_cmd_spread_us",
        "soft_trigger_cmd_ok",
        "soft_trigger_cmd_ret",
        "shape_h",
        "shape_w",
        "shape_c",
    ]

    def __init__(self, sync_cfg: Dict[str, Any], route_order: List[str]):
        self.sync_cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.enabled = _sync_enabled(self.sync_cfg) and _as_bool(self.sync_cfg.get("write_mvs_frame_audit", True), True)
        self.flush = _as_bool(self.sync_cfg.get("mvs_frame_audit_flush", False), False)
        self.root = os.path.abspath(str(self.sync_cfg.get("root", "./sync_ipc") or "./sync_ipc"))
        self.template = str(self.sync_cfg.get("mvs_frame_audit_csv_template", "{root}/mvs_frame_audit_{camera}.csv") or "{root}/mvs_frame_audit_{camera}.csv")
        self.truncate = _as_bool(self.sync_cfg.get("truncate_mvs_frame_audit_on_start", True), True)
        self.files: Dict[str, Any] = {}
        self.writers: Dict[str, Any] = {}
        if self.enabled:
            for cam_id in route_order:
                self._open(str(cam_id))

    def _path_for(self, cam_id: str) -> str:
        path = self.template.format(root=self.root, camera=str(cam_id), cam=str(cam_id), id=str(cam_id))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def _open(self, cam_id: str) -> None:
        if cam_id in self.writers:
            return
        path = self._path_for(cam_id)
        ensure_dir_for_file(path)
        mode = "w" if self.truncate else "a"
        fp = open(path, mode, encoding="utf-8", newline="", buffering=1)
        writer = csv.DictWriter(fp, fieldnames=self.FIELDNAMES)
        if mode == "w" or os.path.getsize(path) == 0:
            writer.writeheader()
        self.files[cam_id] = fp
        self.writers[cam_id] = writer
        print(f"[sync][{cam_id}] mvs frame audit csv -> {path}", flush=True)

    @staticmethod
    def _sync_index_from_meta(mvs_meta: Optional[Dict[str, Any]], mvs_frame_num: int) -> int:
        if isinstance(mvs_meta, dict):
            for key in ("program_sync_index", "sync_index", "sync_index_hint", "mvs_sync_index"):
                val = mvs_meta.get(key, None)
                if val is not None:
                    try:
                        return int(val)
                    except Exception:
                        pass
        try:
            return int(mvs_frame_num)
        except Exception:
            return -1

    def write(self,
              cam_id: str,
              frame: np.ndarray,
              timestamp_us: int,
              fid: int,
              mvs_frame_num: int,
              mvs_meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        cam_id = str(cam_id)
        try:
            self._open(cam_id)
            arr = np.asarray(frame) if frame is not None else None
            if arr is None:
                h = w = c = 0
            elif arr.ndim == 2:
                h, w = arr.shape[:2]
                c = 1
            else:
                h, w = arr.shape[:2]
                c = int(arr.shape[2]) if arr.ndim >= 3 else 1
            meta = mvs_meta if isinstance(mvs_meta, dict) else {}
            row = {
                "write_wall_us": int(now_us()),
                "camera": cam_id,
                "frame_id_local": int(fid),
                "mvs_frame_num": int(mvs_frame_num),
                "sync_index": int(self._sync_index_from_meta(meta, int(mvs_frame_num))),
                "capture_wall_us": int(timestamp_us or 0),
                "receiver_wall_us": int(meta.get("receiver_wall_us", 0) or 0),
                "soft_trigger_tick_wall_us": int(meta.get("soft_trigger_tick_wall_us", 0) or 0),
                "soft_trigger_done_wall_us": int(meta.get("soft_trigger_done_wall_us", 0) or 0),
                "soft_trigger_cmd_wall_us": int(meta.get("soft_trigger_cmd_wall_us", 0) or 0),
                "soft_trigger_cmd_done_wall_us": int(meta.get("soft_trigger_cmd_done_wall_us", 0) or 0),
                "soft_trigger_cmd_spread_us": int(meta.get("soft_trigger_cmd_spread_us", 0) or 0),
                "soft_trigger_cmd_ok": int(bool(meta.get("soft_trigger_cmd_ok", False))),
                "soft_trigger_cmd_ret": int(meta.get("soft_trigger_cmd_ret", 0) or 0),
                "shape_h": int(h),
                "shape_w": int(w),
                "shape_c": int(c),
            }
            self.writers[cam_id].writerow(row)
            if self.flush:
                self.files[cam_id].flush()
        except Exception as exc:
            # Do not let diagnostics break real-time capture.
            if not hasattr(self, "_warned"):
                self._warned = set()
            if cam_id not in self._warned:
                self._warned.add(cam_id)
                print(f"[sync][{cam_id}][WARN] mvs frame audit write failed: {type(exc).__name__}: {exc}", flush=True)

    def close(self) -> None:
        for fp in list(self.files.values()):
            try:
                fp.close()
            except Exception:
                pass
        self.files.clear()
        self.writers.clear()


def _parse_bgr_color(value: Any, default: Tuple[int, int, int] = (0, 255, 255)) -> Tuple[int, int, int]:
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#") and len(s) in (7, 9):
            try:
                r = int(s[1:3], 16)
                g = int(s[3:5], 16)
                b = int(s[5:7], 16)
                return (b, g, r)
            except Exception:
                return default
        named = {
            "cyan": (255, 255, 0),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "white": (255, 255, 255),
        }
        return named.get(s.lower(), default)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(int(max(0, min(255, int(v)))) for v in value[:3])  # type: ignore[return-value]
        except Exception:
            return default
    return default


def _load_shared_frame_reader_class():
    try:
        from latest_frame_shm import SharedFrameReader  # type: ignore
        return SharedFrameReader
    except Exception:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from latest_frame_shm import SharedFrameReader  # type: ignore
        return SharedFrameReader


class EventSharedOverlayManager:
    """Read per-camera Event shared-memory frames and alpha-composite them onto MVS preview frames.

    2026-05 timing fix:
      - The old display path called read_latest() inside apply(), after YOLO inference.
        That could compose an old MVS frame with a newer event overlay frame.
      - The new path lets the main loop call capture_for_mvs_frame() immediately after
        selecting an MVS frame, then reuses that captured/matched event item after YOLO.
      - When sync_match_enable is true, the manager reads event_trigger_{camera}.csv,
        maps MVS program_sync_index -> event_ts_us, and selects the nearest overlay frame
        from a small per-camera background history buffer.
    """

    def __init__(self, cfg: Dict[str, Any], sync_cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.sync_cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.enabled = _as_bool(self.cfg.get("enabled", False), False)
        self.save_path = str(self.cfg.get("_save_path", self.cfg.get("save_path", "")) or "").strip()
        self.base_cfg = {k: copy.deepcopy(v) for k, v in self.cfg.items() if k not in ("per_route", "routes") and not str(k).startswith("_")}
        per = self.cfg.get("per_route", self.cfg.get("routes", {}))
        self.route_overrides: Dict[str, Dict[str, Any]] = copy.deepcopy(per) if isinstance(per, dict) else {}
        self.readers: Dict[str, Any] = {}
        self._reader_locks: Dict[str, threading.Lock] = {}
        # cam_id -> (frame, wall_time_s, event_timestamp_us, shared_frame_id)
        self.last_frames: Dict[str, Tuple[np.ndarray, float, int, int]] = {}
        self.warned: Dict[str, float] = {}
        self.selected_cam: str = ""
        self._printed_shapes: set = set()

        self._history: Dict[str, deque] = {}
        self._history_locks: Dict[str, threading.Lock] = {}
        self._history_threads: Dict[str, threading.Thread] = {}
        self._history_stop = threading.Event()
        self._last_history_frame_id: Dict[str, int] = {}

        # CSV cache: cam_id -> {sync_index: event_ts_us}
        self._trigger_maps: Dict[str, Dict[int, int]] = {}
        self._trigger_map_loaded_wall: Dict[str, float] = {}
        self._trigger_map_paths: Dict[str, str] = {}
        # Incremental CSV reader state. Re-reading event_trigger_*.csv from the
        # beginning every 0.2s becomes expensive during long captures, so we keep
        # the last file offset and only parse newly appended rows. The public cache
        # self._trigger_maps remains a simple {sync_index: event_ts_us} mapping.
        self._trigger_csv_states: Dict[str, Dict[str, Any]] = {}
        self._debug_last_wall: Dict[str, float] = {}

        if self.enabled:
            print(
                "[event_overlay] enabled: source=shared_memory "
                f"name_template={self.cfg.get('name_template', 'event_overlay_latest_{camera}')} "
                f"history_enable={self.cfg.get('history_enable', True)} "
                f"sync_match_enable={self.cfg.get('sync_match_enable', True)}",
                flush=True,
            )
            self.print_help()

    def close(self) -> None:
        self._history_stop.set()
        for th in list(self._history_threads.values()):
            try:
                th.join(timeout=0.5)
            except Exception:
                pass
        self._history_threads.clear()
        for reader in list(self.readers.values()):
            try:
                reader.close()
            except Exception:
                pass
        self.readers.clear()

    def _route_cfg(self, cam_id: str) -> Dict[str, Any]:
        out = copy.deepcopy(self.base_cfg)
        override = self.route_overrides.get(str(cam_id), {})
        if isinstance(override, dict):
            _deep_update(out, copy.deepcopy(override))
        return out

    def _override_for(self, cam_id: str) -> Dict[str, Any]:
        cam_id = str(cam_id)
        if cam_id not in self.route_overrides or not isinstance(self.route_overrides.get(cam_id), dict):
            self.route_overrides[cam_id] = {}
        return self.route_overrides[cam_id]

    def print_help(self) -> None:
        if not self.enabled:
            return
        print(
            "[event_overlay][keys] 1-9/select route, I/K move Y, J/L move X, -/= scale, ,/. black_threshold, "
            "R reset selected, V save to JSON, H help",
            flush=True,
        )

    def select(self, cam_id: str) -> None:
        cam_id = str(cam_id)
        if cam_id:
            self.selected_cam = cam_id
            print(f"[event_overlay] selected route={cam_id} {self.describe(cam_id)}", flush=True)

    def _selected_or_default(self, route_order: List[str]) -> str:
        if self.selected_cam in route_order:
            return self.selected_cam
        if route_order:
            self.select(str(route_order[0]))
            return str(route_order[0])
        return self.selected_cam

    def describe(self, cam_id: str) -> str:
        rcfg = self._route_cfg(cam_id)
        return (
            f"scale={float(rcfg.get('scale', 1.0) or 1.0):.4f} "
            f"x={float(rcfg.get('x', rcfg.get('offset_x', 0)) or 0):.1f} "
            f"y={float(rcfg.get('y', rcfg.get('offset_y', 0)) or 0):.1f} "
            f"black_threshold={float(rcfg.get('black_threshold', 8) or 0):.1f} "
            f"delay_ms={float(rcfg.get('event_overlay_delay_ms', 0.0) or 0.0):.1f}"
        )

    def _set_selected_value(self, cam_id: str, key: str, value: Any) -> None:
        self._override_for(cam_id)[key] = value

    def _adjust_selected(self, cam_id: str, key: str, delta: float, default: float) -> None:
        rcfg = self._route_cfg(cam_id)
        cur = float(rcfg.get(key, default) or default)
        val = cur + float(delta)
        if key == "scale":
            val = max(0.05, min(3.0, val))
        elif key == "black_threshold":
            val = max(0.0, min(255.0, val))
        self._set_selected_value(cam_id, key, round(val, 4) if key == "scale" else round(val, 2))
        print(f"[event_overlay] route={cam_id} {self.describe(cam_id)}", flush=True)

    def reset_selected(self, cam_id: str) -> None:
        self.route_overrides[str(cam_id)] = {"scale": 1.0, "x": 0, "y": 0}
        print(f"[event_overlay] route={cam_id} reset {self.describe(cam_id)}", flush=True)

    def handle_key(self, key: int, route_order: List[str]) -> bool:
        if not self.enabled or key is None or int(key) < 0:
            return False
        key = int(key) & 0xFF
        for idx, cam_id in enumerate(route_order[:9], start=1):
            if key == ord(str(idx)):
                self.select(str(cam_id))
                return True
        cam_id = self._selected_or_default(route_order)
        if not cam_id:
            return False
        if key in (ord("h"), ord("H")):
            self.print_help()
            print(f"[event_overlay] selected route={cam_id} {self.describe(cam_id)}", flush=True)
            return True
        if key in (ord("v"), ord("V")):
            self.save()
            return True
        if key in (ord("r"), ord("R")):
            self.reset_selected(cam_id)
            return True
        move_small = 2.0
        move_big = 10.0
        if key == ord("j"):
            self._adjust_selected(cam_id, "x", -move_small, 0.0)
            return True
        if key == ord("l"):
            self._adjust_selected(cam_id, "x", move_small, 0.0)
            return True
        if key == ord("i"):
            self._adjust_selected(cam_id, "y", -move_small, 0.0)
            return True
        if key == ord("k"):
            self._adjust_selected(cam_id, "y", move_small, 0.0)
            return True
        if key == ord("J"):
            self._adjust_selected(cam_id, "x", -move_big, 0.0)
            return True
        if key == ord("L"):
            self._adjust_selected(cam_id, "x", move_big, 0.0)
            return True
        if key == ord("I"):
            self._adjust_selected(cam_id, "y", -move_big, 0.0)
            return True
        if key == ord("K"):
            self._adjust_selected(cam_id, "y", move_big, 0.0)
            return True
        if key in (ord("-"), ord("_")):
            self._adjust_selected(cam_id, "scale", -0.01, 1.0)
            return True
        if key in (ord("="), ord("+")):
            self._adjust_selected(cam_id, "scale", 0.01, 1.0)
            return True
        if key == ord(","):
            self._adjust_selected(cam_id, "black_threshold", -1.0, 8.0)
            return True
        if key == ord("."):
            self._adjust_selected(cam_id, "black_threshold", 1.0, 8.0)
            return True
        return False

    def save(self) -> None:
        path = self.save_path
        if not path:
            print("[event_overlay][WARN] save path missing; cannot save overlay config", flush=True)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            overlay_cfg = obj.setdefault("event_overlay", {})
            for k, v in self.base_cfg.items():
                overlay_cfg[k] = copy.deepcopy(v)
            overlay_cfg["per_route"] = copy.deepcopy(self.route_overrides)
            overlay_cfg.pop("_save_path", None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"[event_overlay] saved overlay config -> {path}", flush=True)
        except Exception as exc:
            print(f"[event_overlay][WARN] save failed: {type(exc).__name__}: {exc}", flush=True)

    def _warn_once(self, cam_id: str, msg: str, interval_s: float = 5.0) -> None:
        now = time.time()
        last = float(self.warned.get(cam_id, 0.0))
        if now - last >= interval_s:
            self.warned[cam_id] = now
            print(f"[event_overlay][{cam_id}][WARN] {msg}", flush=True)

    def _reader_name(self, cam_id: str, rcfg: Dict[str, Any]) -> str:
        template = str(rcfg.get("name_template", "event_overlay_latest_{camera}") or "event_overlay_latest_{camera}")
        return template.format(camera=str(cam_id), cam=str(cam_id))

    def _reader_for(self, cam_id: str, rcfg: Dict[str, Any]):
        cam_id = str(cam_id)
        if cam_id in self.readers:
            return self.readers[cam_id]
        name = self._reader_name(cam_id, rcfg)
        try:
            reader_cls = _load_shared_frame_reader_class()
            reader = reader_cls(name)
            self.readers[cam_id] = reader
            print(f"[event_overlay][{cam_id}] shared frame opened: {name}", flush=True)
            return reader
        except Exception as exc:
            self._warn_once(cam_id, f"cannot open shared frame {name}: {type(exc).__name__}: {exc}")
            return None

    def _history_enabled(self, rcfg: Dict[str, Any]) -> bool:
        return _as_bool(rcfg.get("history_enable", True), True)

    def _reader_lock_for(self, cam_id: str) -> threading.Lock:
        cam_id = str(cam_id)
        lock = self._reader_locks.get(cam_id)
        if lock is None:
            lock = threading.Lock()
            self._reader_locks[cam_id] = lock
        return lock

    def _history_lock_for(self, cam_id: str) -> threading.Lock:
        cam_id = str(cam_id)
        lock = self._history_locks.get(cam_id)
        if lock is None:
            lock = threading.Lock()
            self._history_locks[cam_id] = lock
        return lock

    def _append_history_item(self, cam_id: str, item: Dict[str, Any], rcfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        frame = item.get("frame") if isinstance(item, dict) else None
        if frame is None:
            return None
        cam_id = str(cam_id)
        frame_id = int(item.get("frame_id", 0) or 0)
        ts = int(item.get("timestamp_us", 0) or 0)
        if ts <= 0:
            return None
        if int(self._last_history_frame_id.get(cam_id, -1)) == frame_id:
            return None
        self._last_history_frame_id[cam_id] = frame_id
        hist_item = {
            "frame": np.asarray(frame).copy(),
            "timestamp_us": int(ts),
            "frame_id": int(frame_id),
            "recv_wall_s": float(time.time()),
            "source": "history_read",
        }
        max_age_us = int(float(rcfg.get("history_max_age_ms", 500.0) or 500.0) * 1000.0)
        max_frames = int(rcfg.get("history_max_frames", 160) or 160)
        lock = self._history_lock_for(cam_id)
        with lock:
            q = self._history.setdefault(cam_id, deque())
            q.append(hist_item)
            cutoff_ts = int(ts) - max_age_us
            while q and int(q[0].get("timestamp_us", 0)) < cutoff_ts:
                q.popleft()
            while len(q) > max_frames:
                q.popleft()
        self.last_frames[cam_id] = (hist_item["frame"], hist_item["recv_wall_s"], int(ts), int(frame_id))
        return hist_item

    def _history_loop(self, cam_id: str) -> None:
        cam_id = str(cam_id)
        while not self._history_stop.is_set():
            rcfg = self._route_cfg(cam_id)
            reader = self._reader_for(cam_id, rcfg)
            if reader is None:
                time.sleep(0.2)
                continue
            try:
                with self._reader_lock_for(cam_id):
                    item = reader.read_latest()
            except Exception as exc:
                self._warn_once(cam_id, f"read shared frame failed in history loop: {type(exc).__name__}: {exc}")
                item = None
            if item is not None:
                self._append_history_item(cam_id, item, rcfg)
            poll_ms = float(rcfg.get("history_poll_interval_ms", 2.0) or 2.0)
            time.sleep(max(0.0005, poll_ms / 1000.0))

    def _ensure_history_thread(self, cam_id: str, rcfg: Dict[str, Any]) -> None:
        if not self.enabled or not self._history_enabled(rcfg):
            return
        cam_id = str(cam_id)
        th = self._history_threads.get(cam_id)
        if th is not None and th.is_alive():
            return
        self._history_stop.clear()
        th = threading.Thread(target=self._history_loop, args=(cam_id,), daemon=True, name=f"event_overlay_history_{cam_id}")
        self._history_threads[cam_id] = th
        th.start()
        print(f"[event_overlay][{cam_id}] history reader started", flush=True)

    def start_history(self, route_order: List[str]) -> None:
        if not self.enabled:
            return
        for cam_id in route_order:
            rcfg = self._route_cfg(str(cam_id))
            if self._history_enabled(rcfg):
                self._ensure_history_thread(str(cam_id), rcfg)

    def _read_event_item_now(self, cam_id: str, rcfg: Dict[str, Any], allow_reuse: bool = True) -> Optional[Dict[str, Any]]:
        reader = self._reader_for(cam_id, rcfg)
        if reader is not None:
            try:
                with self._reader_lock_for(cam_id):
                    item = reader.read_latest()
            except Exception as exc:
                self._warn_once(cam_id, f"read shared frame failed: {type(exc).__name__}: {exc}")
                item = None
            if item is not None and item.get("frame") is not None:
                hist_item = self._append_history_item(cam_id, item, rcfg)
                frame = np.asarray(item["frame"]).copy()
                ts = int(item.get("timestamp_us", 0) or 0)
                frame_id = int(item.get("frame_id", 0) or 0)
                out = {
                    "frame": frame,
                    "timestamp_us": ts,
                    "frame_id": frame_id,
                    "recv_wall_s": float(time.time()),
                    "source": "latched_latest",
                }
                if hist_item is not None:
                    out["history_frame_id"] = hist_item.get("frame_id")
                self.last_frames[str(cam_id)] = (frame, out["recv_wall_s"], ts, frame_id)
                return out

        if allow_reuse and _as_bool(rcfg.get("reuse_last_frame", True), True):
            cached = self.last_frames.get(str(cam_id))
            if cached is not None:
                max_age_ms = float(rcfg.get("max_stale_ms", 250.0) or 250.0)
                age_ms = (time.time() - float(cached[1])) * 1000.0
                if age_ms <= max_age_ms:
                    return {
                        "frame": cached[0].copy(),
                        "timestamp_us": int(cached[2]),
                        "frame_id": int(cached[3]),
                        "recv_wall_s": float(cached[1]),
                        "source": "reuse_last_frame",
                        "reuse_age_ms": float(age_ms),
                    }
        return None

    def _trigger_csv_path(self, cam_id: str) -> str:
        root = str((self.sync_cfg or {}).get("root", "./sync_ipc") or "./sync_ipc")
        root = os.path.abspath(root)
        template = str(
            (self.sync_cfg or {}).get(
                "event_trigger_csv_template",
                (self.sync_cfg or {}).get("event_trigger_template", "{root}/event_trigger_{camera}.csv"),
            ) or "{root}/event_trigger_{camera}.csv"
        )
        path = template.format(root=root, camera=str(cam_id), cam=str(cam_id))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def _parse_trigger_csv_row(self, row: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        try:
            sync_index = int(str(row.get("sync_index", "")).strip())
        except Exception:
            return None
        if sync_index < 0:
            return None
        ts_val = row.get("event_ts_us", row.get("timestamp_us", row.get("timestamp", "0")))
        try:
            event_ts_us = int(float(str(ts_val).strip()))
        except Exception:
            return None
        if event_ts_us <= 0:
            return None
        return int(sync_index), int(event_ts_us)

    def _trim_trigger_map(self, cam_id: str, mapping: Dict[int, int]) -> None:
        max_entries = int(self.cfg.get("trigger_csv_max_entries", 5000) or 5000)
        if max_entries <= 0 or len(mapping) <= max_entries:
            return
        # sync_index is monotonic in event_trigger_*.csv; keep only recent rows.
        remove_count = len(mapping) - max_entries
        for key in sorted(mapping.keys())[:remove_count]:
            mapping.pop(key, None)

    def _reload_trigger_map_full(self, cam_id: str, path: str, now: float) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        try:
            import csv
            header = None
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                for row in reader:
                    parsed = self._parse_trigger_csv_row(row)
                    if parsed is None:
                        continue
                    sync_index, event_ts_us = parsed
                    mapping[sync_index] = event_ts_us
            self._trim_trigger_map(cam_id, mapping)
            self._trigger_maps[cam_id] = mapping
            self._trigger_map_loaded_wall[cam_id] = now
            self._trigger_map_paths[cam_id] = path
            try:
                pos = os.path.getsize(path)
            except Exception:
                pos = 0
            self._trigger_csv_states[cam_id] = {"path": path, "pos": pos, "header": header}
        except FileNotFoundError:
            self._warn_once(cam_id, f"trigger CSV not found: {path}")
            self._trigger_maps.setdefault(cam_id, {})
            self._trigger_map_loaded_wall[cam_id] = now
            self._trigger_map_paths[cam_id] = path
            self._trigger_csv_states[cam_id] = {"path": path, "pos": 0, "header": None}
        except Exception as exc:
            self._warn_once(cam_id, f"load trigger CSV failed: {type(exc).__name__}: {exc}; path={path}")
            self._trigger_maps.setdefault(cam_id, {})
            self._trigger_map_loaded_wall[cam_id] = now
            self._trigger_map_paths[cam_id] = path
        return self._trigger_maps.get(cam_id, {})

    def _reload_trigger_map_if_needed(self, cam_id: str, force: bool = False) -> Dict[int, int]:
        cam_id = str(cam_id)
        path = self._trigger_csv_path(cam_id)
        reload_interval_s = float(self.cfg.get("trigger_csv_reload_interval_s", 0.2) or 0.2)
        now = time.time()
        old_path = self._trigger_map_paths.get(cam_id)
        if (not force and old_path == path and cam_id in self._trigger_maps and
                now - float(self._trigger_map_loaded_wall.get(cam_id, 0.0)) < reload_interval_s):
            return self._trigger_maps.get(cam_id, {})

        if not _as_bool(self.cfg.get("trigger_csv_incremental", True), True):
            return self._reload_trigger_map_full(cam_id, path, now)

        mapping = self._trigger_maps.setdefault(cam_id, {})
        try:
            import csv
            size = os.path.getsize(path)
            state = self._trigger_csv_states.get(cam_id)
            # New file, changed path, or truncation caused by a new run: reset.
            if state is None or state.get("path") != path or size < int(state.get("pos", 0) or 0):
                state = {"path": path, "pos": 0, "header": None}
                self._trigger_csv_states[cam_id] = state
                mapping.clear()

            with open(path, "r", encoding="utf-8", newline="") as f:
                pos = int(state.get("pos", 0) or 0)
                header = state.get("header")
                if pos <= 0 or not header:
                    f.seek(0)
                    header_line = f.readline()
                    if not header_line:
                        state["pos"] = f.tell()
                        self._trigger_map_loaded_wall[cam_id] = now
                        self._trigger_map_paths[cam_id] = path
                        return mapping
                    header = next(csv.reader([header_line]))
                    state["header"] = header
                    state["pos"] = f.tell()
                else:
                    f.seek(pos)

                while True:
                    line_start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    # Avoid consuming a row while another process is still writing it.
                    if not line.endswith("\n"):
                        state["pos"] = line_start
                        break
                    try:
                        values = next(csv.reader([line]))
                    except Exception:
                        state["pos"] = f.tell()
                        continue
                    if not values:
                        state["pos"] = f.tell()
                        continue
                    row = {str(header[i]): values[i] if i < len(values) else "" for i in range(len(header))}
                    parsed = self._parse_trigger_csv_row(row)
                    if parsed is not None:
                        sync_index, event_ts_us = parsed
                        mapping[sync_index] = event_ts_us
                    state["pos"] = f.tell()

            self._trim_trigger_map(cam_id, mapping)
            self._trigger_map_loaded_wall[cam_id] = now
            self._trigger_map_paths[cam_id] = path
        except FileNotFoundError:
            self._warn_once(cam_id, f"trigger CSV not found: {path}")
            self._trigger_maps.setdefault(cam_id, {})
            self._trigger_map_loaded_wall[cam_id] = now
            self._trigger_map_paths[cam_id] = path
            self._trigger_csv_states[cam_id] = {"path": path, "pos": 0, "header": None}
        except Exception as exc:
            self._warn_once(cam_id, f"load trigger CSV incrementally failed: {type(exc).__name__}: {exc}; path={path}")
            # Fallback once to full read so a bad offset/header cannot permanently break matching.
            return self._reload_trigger_map_full(cam_id, path, now)
        return self._trigger_maps.get(cam_id, {})

    def _sync_index_for_mvs(self, cam_id: str, mvs_frame_num: int, mvs_meta: Optional[Dict[str, Any]], rcfg: Dict[str, Any]) -> Optional[int]:
        if isinstance(mvs_meta, dict):
            val = mvs_meta.get("program_sync_index", None)
            if val is not None and str(val) != "":
                try:
                    return int(val)
                except Exception:
                    pass
        offset = int(rcfg.get("sync_index_offset", (self.sync_cfg or {}).get("sync_index_offset", -1)) or -1)
        if int(mvs_frame_num) > 0:
            return int(mvs_frame_num) + offset
        return None

    def _target_event_ts_for_mvs(self, cam_id: str, mvs_frame_num: int, mvs_meta: Optional[Dict[str, Any]], rcfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        sync_index = self._sync_index_for_mvs(cam_id, mvs_frame_num, mvs_meta, rcfg)
        if sync_index is None:
            return None, None
        mapping = self._reload_trigger_map_if_needed(cam_id, force=False)
        event_ts = mapping.get(int(sync_index))
        if event_ts is None:
            # The CSV is written live; if the first cache read was just before this row arrived, retry once.
            mapping = self._reload_trigger_map_if_needed(cam_id, force=True)
            event_ts = mapping.get(int(sync_index))
        if event_ts is None:
            return int(sync_index), None
        delay_ms = float(rcfg.get("event_overlay_delay_ms", 0.0) or 0.0)
        offset_ms = float(rcfg.get("event_overlay_time_offset_ms", 0.0) or 0.0)
        target = int(event_ts + (offset_ms - delay_ms) * 1000.0)
        return int(sync_index), int(target)

    def _nearest_history_item(self, cam_id: str, target_ts_us: int, rcfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cam_id = str(cam_id)
        max_diff_us = int(float(rcfg.get("event_overlay_max_match_diff_ms", 120.0) or 120.0) * 1000.0)
        lock = self._history_lock_for(cam_id)
        with lock:
            q = list(self._history.get(cam_id, deque()))
        if not q:
            return None
        best = min(q, key=lambda it: abs(int(it.get("timestamp_us", 0)) - int(target_ts_us)))
        diff_us = int(best.get("timestamp_us", 0)) - int(target_ts_us)
        if abs(diff_us) > max_diff_us:
            return None
        out = {
            "frame": np.asarray(best["frame"]).copy(),
            "timestamp_us": int(best.get("timestamp_us", 0)),
            "frame_id": int(best.get("frame_id", 0)),
            "recv_wall_s": float(best.get("recv_wall_s", 0.0)),
            "source": "history_nearest",
            "target_event_ts_us": int(target_ts_us),
            "match_diff_us": int(diff_us),
        }
        return out

    def _latest_history_item(self, cam_id: str) -> Optional[Dict[str, Any]]:
        lock = self._history_lock_for(cam_id)
        with lock:
            q = self._history.get(str(cam_id), deque())
            if not q:
                return None
            best = q[-1]
        return {
            "frame": np.asarray(best["frame"]).copy(),
            "timestamp_us": int(best.get("timestamp_us", 0)),
            "frame_id": int(best.get("frame_id", 0)),
            "recv_wall_s": float(best.get("recv_wall_s", 0.0)),
            "source": "history_latest",
        }

    def _history_item_at_or_before(self, cam_id: str, target_ts_us: int) -> Optional[Dict[str, Any]]:
        """Return the newest event-overlay history item whose event timestamp is <= target_ts_us."""
        lock = self._history_lock_for(cam_id)
        with lock:
            q = list(self._history.get(str(cam_id), deque()))
        if not q:
            return None
        best = None
        for it in reversed(q):
            try:
                ts = int(it.get("timestamp_us", 0) or 0)
            except Exception:
                continue
            if ts <= int(target_ts_us):
                best = it
                break
        if best is None:
            best = q[0]
        return {
            "frame": np.asarray(best["frame"]).copy(),
            "timestamp_us": int(best.get("timestamp_us", 0)),
            "frame_id": int(best.get("frame_id", 0)),
            "recv_wall_s": float(best.get("recv_wall_s", 0.0)),
            "source": "history_at_or_before",
            "target_display_event_ts_us": int(target_ts_us),
            "display_delay_diff_us": int(best.get("timestamp_us", 0)) - int(target_ts_us),
        }

    def capture_event_driven_item(self, cam_id: str, display_delay_ms: float = 0.0) -> Optional[Dict[str, Any]]:
        """Return an event overlay item for 250fps event-driven fusion.

        The history reader already polls the shared-memory publisher in the background.
        This method normally reads from that history instead of calling read_latest() for
        every display tick. A small display_delay_ms gives the MVS receiver time to deliver
        the anchor frame for the next 40ms period before the event overlay is displayed.
        """
        if not self.enabled:
            return None
        cam_id = str(cam_id)
        rcfg = self._route_cfg(cam_id)
        if not _as_bool(rcfg.get("enabled", True), True):
            return None
        if self._history_enabled(rcfg):
            self._ensure_history_thread(cam_id, rcfg)
        latest = self._latest_history_item(cam_id)
        if latest is not None:
            delay_us = int(max(0.0, float(display_delay_ms or 0.0)) * 1000.0)
            if delay_us > 0:
                target_ts = int(latest.get("timestamp_us", 0) or 0) - delay_us
                delayed = self._history_item_at_or_before(cam_id, target_ts)
                if delayed is not None:
                    delayed["source"] = "event_driven_delayed_history"
                    delayed["display_delay_ms"] = float(display_delay_ms or 0.0)
                    return delayed
            latest["source"] = "event_driven_history_latest"
            latest["display_delay_ms"] = float(display_delay_ms or 0.0)
            return latest
        item = self._read_event_item_now(cam_id, rcfg, allow_reuse=True)
        if item is not None:
            item["source"] = "event_driven_read_now"
            item["display_delay_ms"] = float(display_delay_ms or 0.0)
        return item

    def resolve_mvs_anchor_time(self,
                                cam_id: str,
                                mvs_ts_us: int = 0,
                                mvs_frame_num: int = 0,
                                mvs_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolve the event-clock timestamp that should anchor this MVS frame.

        Preferred source is event_trigger_{camera}.csv: program_sync_index -> event_ts_us.
        That keeps MVS and event in the event camera time domain. If CSV matching is not
        ready yet, return a fallback timestamp so preview can still run, but mark it as
        non-strict.
        """
        cam_id = str(cam_id)
        rcfg = self._route_cfg(cam_id)
        sync_index, target_event_ts = self._target_event_ts_for_mvs(cam_id, int(mvs_frame_num or 0), mvs_meta, rcfg)
        if target_event_ts is not None:
            return {
                "anchor_event_ts_us": int(target_event_ts),
                "mvs_sync_index": int(sync_index) if sync_index is not None else None,
                "source": "trigger_csv",
                "strict": True,
            }
        latest = self._latest_history_item(cam_id)
        if latest is not None and int(latest.get("timestamp_us", 0) or 0) > 0:
            return {
                "anchor_event_ts_us": int(latest.get("timestamp_us", 0) or 0),
                "mvs_sync_index": int(sync_index) if sync_index is not None else None,
                "source": "latest_event_fallback",
                "strict": False,
            }
        return {
            "anchor_event_ts_us": int(mvs_ts_us or 0),
            "mvs_sync_index": int(sync_index) if sync_index is not None else None,
            "source": "mvs_wall_clock_fallback",
            "strict": False,
        }

    def _debug_event_item(self, cam_id: str, item: Optional[Dict[str, Any]], rcfg: Dict[str, Any], mvs_frame_num: int = 0, mvs_sync_index: Optional[int] = None) -> None:
        if not _as_bool(rcfg.get("debug_timing", False), False):
            return
        now = time.time()
        interval_s = float(rcfg.get("debug_interval_s", 1.0) or 1.0)
        last = float(self._debug_last_wall.get(str(cam_id), 0.0))
        if now - last < interval_s:
            return
        self._debug_last_wall[str(cam_id)] = now
        if item is None:
            print(f"[event_overlay][{cam_id}][timing] no event item for MVS={mvs_frame_num} sync={mvs_sync_index}", flush=True)
            return
        diff = item.get("match_diff_us", None)
        diff_text = "-" if diff is None else f"{float(diff) / 1000.0:.3f}ms"
        target = item.get("target_event_ts_us", "-")
        age_wall = (now - float(item.get("recv_wall_s", now))) * 1000.0 if item.get("recv_wall_s") else 0.0
        print(
            f"[event_overlay][{cam_id}][timing] source={item.get('source')} "
            f"MVS={mvs_frame_num} sync={mvs_sync_index} "
            f"target_event_ts={target} event_ts={item.get('timestamp_us')} "
            f"diff={diff_text} event_frame_id={item.get('frame_id')} "
            f"wall_age={age_wall:.1f}ms",
            flush=True,
        )

    def capture_for_mvs_frame(self,
                              cam_id: str,
                              mvs_ts_us: int = 0,
                              mvs_frame_num: int = 0,
                              mvs_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Capture or timestamp-match one event overlay item for exactly one selected MVS frame.

        The returned item must travel with the MVS frame through YOLO, and later be passed
        to apply_item(). This prevents YOLO inference time from moving the event layer into
        the future relative to that MVS frame.
        """
        if not self.enabled:
            return None
        cam_id = str(cam_id)
        rcfg = self._route_cfg(cam_id)
        if not _as_bool(rcfg.get("enabled", True), True):
            return None

        if self._history_enabled(rcfg):
            self._ensure_history_thread(cam_id, rcfg)

        selected: Optional[Dict[str, Any]] = None
        sync_index: Optional[int] = None
        if _as_bool(rcfg.get("sync_match_enable", True), True):
            sync_index, target_event_ts = self._target_event_ts_for_mvs(cam_id, int(mvs_frame_num), mvs_meta, rcfg)
            if target_event_ts is not None:
                selected = self._nearest_history_item(cam_id, int(target_event_ts), rcfg)
                if selected is not None:
                    selected["source"] = "sync_csv_history"
                    selected["mvs_sync_index"] = sync_index
                    selected["mvs_frame_num"] = int(mvs_frame_num)
                elif not _as_bool(rcfg.get("fallback_to_latched_latest", True), True):
                    self._debug_event_item(cam_id, None, rcfg, int(mvs_frame_num), sync_index)
                    return None

        # If strict CSV/history match was not possible, still latch at MVS frame selection time.
        # This is the minimal timing fix: no read_latest() after YOLO inference.
        if selected is None:
            delay_ms = float(rcfg.get("event_overlay_delay_ms", 0.0) or 0.0)
            offset_ms = float(rcfg.get("event_overlay_time_offset_ms", 0.0) or 0.0)
            latest = self._latest_history_item(cam_id)
            if latest is not None and (abs(delay_ms) > 1e-6 or abs(offset_ms) > 1e-6):
                target_ts = int(int(latest.get("timestamp_us", 0)) + (offset_ms - delay_ms) * 1000.0)
                selected = self._nearest_history_item(cam_id, target_ts, rcfg)
                if selected is not None:
                    selected["source"] = "delay_history"
            if selected is None:
                selected = self._read_event_item_now(cam_id, rcfg, allow_reuse=True)
                if selected is not None:
                    selected["source"] = str(selected.get("source", "latched_latest"))

        if selected is not None:
            selected["mvs_capture_wall_us"] = int(mvs_ts_us or 0)
            selected["mvs_frame_num"] = int(mvs_frame_num or 0)
            if sync_index is not None:
                selected["mvs_sync_index"] = int(sync_index)
        self._debug_event_item(cam_id, selected, rcfg, int(mvs_frame_num), sync_index)
        return selected

    # Backward-compatible name for the previous code path.
    def read_for_camera(self, cam_id: str) -> Optional[Dict[str, Any]]:
        return self.capture_for_mvs_frame(cam_id)

    def _get_homography(self, cam_id: str, rcfg: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return event->MVS homography for this route, if configured.

        Supported config forms:
          1) "homography_event_to_mvs": [[...], [...], [...]]
          2) "homographies_event_to_mvs": {
                 "ground": [[...], [...], [...]],
                 "flight": [[...], [...], [...]]
             },
             "active_homography": "flight"
        """
        H_value = None

        hs = rcfg.get("homographies_event_to_mvs")
        if isinstance(hs, dict) and hs:
            active = str(rcfg.get("active_homography", "flight") or "flight")
            H_value = hs.get(active)
            if H_value is None and len(hs) == 1:
                H_value = next(iter(hs.values()))
            if H_value is None:
                self._warn_once(cam_id, f"active_homography={active!r} not found in homographies_event_to_mvs")
                return None
        else:
            H_value = rcfg.get("homography_event_to_mvs")

        if H_value is None:
            return None

        try:
            H = np.asarray(H_value, dtype=np.float64)
        except Exception as exc:
            self._warn_once(cam_id, f"invalid homography_event_to_mvs: {type(exc).__name__}: {exc}")
            return None

        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            self._warn_once(cam_id, f"invalid homography_event_to_mvs shape/value: shape={H.shape}")
            return None
        return H

    def _build_layer_and_alpha(self, src: np.ndarray, rcfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if src.ndim == 3 and src.shape[2] == 1:
            gray = src[:, :, 0]
            layer_bgr = None
        elif src.ndim == 2:
            gray = src
            layer_bgr = None
        else:
            layer_bgr = src[:, :, :3].astype(np.uint8)
            gray = np.max(layer_bgr, axis=2).astype(np.uint8)

        threshold = float(rcfg.get("black_threshold", rcfg.get("chroma_black_threshold", 8)) or 0)
        denom = max(1.0, 255.0 - threshold)
        alpha = (gray.astype(np.float32) - threshold) / denom
        alpha = np.clip(alpha, 0.0, 1.0)
        if _as_bool(rcfg.get("binary_mask", False), False):
            alpha = (gray > threshold).astype(np.float32)

        blur = int(rcfg.get("mask_blur", 0) or 0)
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)

        opacity = float(rcfg.get("opacity", rcfg.get("alpha", 1.0)) or 1.0)
        alpha = np.clip(alpha * opacity, 0.0, 1.0)

        if layer_bgr is None or _as_bool(rcfg.get("colorize", True), True):
            color = np.asarray(_parse_bgr_color(rcfg.get("color", "yellow")), dtype=np.float32)
            intensity = gray.astype(np.float32) / 255.0
            layer = np.clip(intensity[:, :, None] * color[None, None, :], 0, 255).astype(np.uint8)
        else:
            layer = layer_bgr
        return layer, alpha

    def _blend_event_frame(self, cam_id: str, frame_bgr: np.ndarray, event_frame: np.ndarray, rcfg: Dict[str, Any]) -> np.ndarray:
        event_h, event_w = event_frame.shape[:2]
        layer, alpha = self._build_layer_and_alpha(event_frame, rcfg)

        base_h, base_w = frame_bgr.shape[:2]

        H = self._get_homography(str(cam_id), rcfg)
        if H is not None:
            warped_layer = cv2.warpPerspective(
                layer,
                H,
                (base_w, base_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped_alpha = cv2.warpPerspective(
                alpha,
                H,
                (base_w, base_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_alpha = np.clip(warped_alpha, 0.0, 1.0)

            shape_key = (
                str(cam_id),
                "homography",
                int(event_w),
                int(event_h),
                int(base_w),
                int(base_h),
                str(rcfg.get("active_homography", "")),
            )
            if shape_key not in self._printed_shapes:
                self._printed_shapes.add(shape_key)
                print(
                    f"[event_overlay][{cam_id}] sizes: event={event_w}x{event_h} "
                    f"mvs={base_w}x{base_h} mode=homography "
                    f"active={rcfg.get('active_homography', rcfg.get('plane', ''))}",
                    flush=True,
                )

            out = frame_bgr.copy()
            roi = out.astype(np.float32)
            lay = warped_layer.astype(np.float32)
            a = warped_alpha.astype(np.float32)[:, :, None]
            blended = lay * a + roi * (1.0 - a)
            out[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
            return out

        scale = float(rcfg.get("scale", 1.0) or 1.0)
        scale = max(0.01, scale)
        if abs(scale - 1.0) > 1e-6:
            new_w = max(1, int(round(layer.shape[1] * scale)))
            new_h = max(1, int(round(layer.shape[0] * scale)))
            layer = cv2.resize(layer, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        layer_h, layer_w = layer.shape[:2]
        x = int(round(float(rcfg.get("x", rcfg.get("offset_x", 0)) or 0)))
        y = int(round(float(rcfg.get("y", rcfg.get("offset_y", 0)) or 0)))
        mode = str(rcfg.get("position_mode", "center_offset") or "center_offset").lower()
        shape_key = (
            str(cam_id),
            int(event_w),
            int(event_h),
            int(base_w),
            int(base_h),
            int(layer_w),
            int(layer_h),
            round(scale, 4),
            mode,
            int(x),
            int(y),
        )
        if shape_key not in self._printed_shapes:
            self._printed_shapes.add(shape_key)
            print(
                f"[event_overlay][{cam_id}] sizes: event={event_w}x{event_h} "
                f"mvs={base_w}x{base_h} layer={layer_w}x{layer_h} "
                f"scale={scale:.4f} position_mode={mode} x={x} y={y}",
                flush=True,
            )
        if mode in ("top_left", "topleft", "xy"):
            x0, y0 = x, y
        else:
            x0 = int(round((base_w - layer_w) * 0.5 + x))
            y0 = int(round((base_h - layer_h) * 0.5 + y))

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(base_w, x0 + layer_w)
        dst_y1 = min(base_h, y0 + layer_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return frame_bgr

        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        out = frame_bgr.copy()
        roi = out[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
        lay = layer[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
        a = alpha[src_y0:src_y1, src_x0:src_x1].astype(np.float32)[:, :, None]
        blended = lay * a + roi * (1.0 - a)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def apply_item(self, cam_id: str, frame_bgr: np.ndarray, event_item: Optional[Dict[str, Any]]) -> np.ndarray:
        if not self.enabled or frame_bgr is None:
            return frame_bgr
        rcfg = self._route_cfg(cam_id)
        if not _as_bool(rcfg.get("enabled", True), True):
            return frame_bgr
        if event_item is None or event_item.get("frame") is None:
            return frame_bgr
        return self._blend_event_frame(str(cam_id), frame_bgr, np.asarray(event_item["frame"]), rcfg)

    def apply(self, cam_id: str, frame_bgr: np.ndarray) -> np.ndarray:
        # Compatibility path: still works, but the main YOLO display loop now uses
        # capture_for_mvs_frame()+apply_item() so it does not read latest after inference.
        if not self.enabled or frame_bgr is None:
            return frame_bgr
        rcfg = self._route_cfg(cam_id)
        if not _as_bool(rcfg.get("enabled", True), True):
            return frame_bgr
        event_item = self._read_event_item_now(str(cam_id), rcfg, allow_reuse=True)
        return self.apply_item(str(cam_id), frame_bgr, event_item)


class HikFullViewSource(BaseFrameSource):
    """海康完整视野取流类。

    关键点：
      - 不把 Width/Height 当成缩放使用；Width/Height 在工业相机里通常是 ROI 裁剪。
      - 当 force_full_sensor=true 且 capture_width/capture_height 为 0 时，主动把 OffsetX/OffsetY 置 0，
        并把 Width/Height 设置到 SDK 可读到的最大值，尽量恢复完整传感器视野。
      - 每帧用 MV_CC_GetImageBuffer 取流；若相机实际输出 BGR8_Packed，则直接包 BGR 图像并跳过 MV_CC_ConvertPixelType/cvtColor；否则旧转换兜底。
    """

    def __init__(self,
                 serial: str = "",
                 mvs_sdk_path: str = "/opt/MVS/Samples/64/Python/MvImport",
                 timeout_ms: int = 1000,
                 width: int = 0,
                 height: int = 0,
                 fps: float = 25.0,
                 force_full_sensor: bool = True,
                 prefer_bgr8: bool = False,
                 sync_cfg: Optional[Dict[str, Any]] = None):
        self.serial = str(serial or "").strip()
        self.mvs_sdk_path = str(mvs_sdk_path or "/opt/MVS/Samples/64/Python/MvImport")
        self.timeout_ms = int(timeout_ms)
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.fps = float(fps or 0.0)
        self.force_full_sensor = bool(force_full_sensor)
        # 安全版：是否请求相机端 BGR8_Packed 仍由 JSON 的 prefer_bgr8 控制。
        # 这样可以保留 BGR8 fast path，但不会在所有相机上无条件放大带宽/改变 PixelFormat。
        # 真正是否成功以后续每帧 stFrameInfo.enPixelType 为准；非 BGR8 仍会走 SDK 转换兜底。
        self.prefer_bgr8 = bool(prefer_bgr8)
        self.prefer_bgr8_from_config = bool(prefer_bgr8)
        self.sync_cfg = sync_cfg or {}
        self.last_mvs_frame_num = 0
        self.last_mvs_frame_info: Dict[str, Any] = {}
        # 同型号相机专用版：不再提供 Decimation/Binning 降采样配置。
        # 但会在启动时主动把相机里可能残留的 Decimation/Binning 复位为 1，避免旧配置影响画幅。
        self.source_name = f"hik_{self.serial or 'auto'}"
        self.cam = None
        self.mvs = None
        self._first_shape_printed = False
        self._import_mvs()
        self._open_device_no_start()
        self._configure_before_start()
        self._start_grabbing()

    def _import_mvs(self) -> None:
        if self.mvs_sdk_path and self.mvs_sdk_path not in sys.path:
            sys.path.append(self.mvs_sdk_path)
        try:
            self.mvs = importlib.import_module("MvCameraControl_class")
        except Exception as e:
            raise RuntimeError(f"导入海康 MVS Python wrapper 失败：{self.mvs_sdk_path}/MvCameraControl_class.py\n原始错误: {e}")

    @staticmethod
    def _decode_c_array(v: Any) -> str:
        try:
            b = bytes(v)
        except Exception:
            try:
                b = bytes(bytearray(v))
            except Exception:
                return ""
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

    def _device_serial(self, info: Any) -> str:
        mvs = self.mvs
        try:
            if info.nTLayerType == getattr(mvs, "MV_GIGE_DEVICE"):
                return self._decode_c_array(info.SpecialInfo.stGigEInfo.chSerialNumber)
            if info.nTLayerType == getattr(mvs, "MV_USB_DEVICE"):
                return self._decode_c_array(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        except Exception:
            pass
        return ""

    def _open_device_no_start(self) -> None:
        mvs = self.mvs
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        tlayer_type = getattr(mvs, "MV_GIGE_DEVICE") | getattr(mvs, "MV_USB_DEVICE")
        ret = mvs.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0 or int(device_list.nDeviceNum) <= 0:
            raise RuntimeError(f"未枚举到海康工业相机设备，ret=0x{ret:x}")
        selected_idx = -1
        found: List[Tuple[int, str]] = []
        for i in range(int(device_list.nDeviceNum)):
            info = ctypes.cast(device_list.pDeviceInfo[i], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            sn = self._device_serial(info)
            found.append((i, sn))
            if self.serial and sn == self.serial:
                selected_idx = i
        if selected_idx < 0:
            if self.serial:
                raise RuntimeError(f"没有找到指定序列号相机: {self.serial}，当前枚举到: {found}")
            selected_idx = 0
        info = ctypes.cast(device_list.pDeviceInfo[selected_idx], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
        sn = self._device_serial(info)
        self.source_name = f"hik_{sn or selected_idx}"
        print(f"[HIK][direct] selected index={selected_idx}, serial={sn or '(unknown)'}")
        self.cam = mvs.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:x}")
        access = getattr(mvs, "MV_ACCESS_Exclusive", 1)
        ret = self.cam.MV_CC_OpenDevice(access, 0)
        if ret != 0:
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:x}")

    def _get_int_param(self, name: str) -> Optional[Any]:
        mvs = self.mvs
        for cls_name, fn_name in [
            ("MVCC_INTVALUE_EX", "MV_CC_GetIntValueEx"),
            ("MVCC_INTVALUE", "MV_CC_GetIntValue"),
        ]:
            cls = getattr(mvs, cls_name, None)
            fn = getattr(self.cam, fn_name, None)
            if cls is None or fn is None:
                continue
            try:
                param = cls()
                ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
                ret = fn(name, param)
                if ret == 0:
                    return param
            except Exception:
                continue
        return None

    @staticmethod
    def _param_value(param: Any, names: List[str], default: Optional[int] = None) -> Optional[int]:
        for n in names:
            if hasattr(param, n):
                try:
                    return int(getattr(param, n))
                except Exception:
                    pass
        return default

    def _set_int_silent(self, name: str, value: int, print_ok: bool = True) -> bool:
        try:
            ret = self.cam.MV_CC_SetIntValue(name, int(value))
            if ret == 0:
                if print_ok:
                    print(f"[HIK][direct] set {name}={int(value)}")
                return True
            return False
        except Exception:
            return False

    def _set_int_or_enum(self, name: str, value: int) -> bool:
        """兼容不同海康型号：有些节点是 Integer，有些节点是 Enumeration。"""
        value = int(value)
        try:
            ret = self.cam.MV_CC_SetIntValue(name, value)
            if ret == 0:
                print(f"[HIK][direct] set {name}={value} by Int")
                return True
        except Exception:
            pass
        try:
            ret = self.cam.MV_CC_SetEnumValue(name, value)
            if ret == 0:
                print(f"[HIK][direct] set {name}={value} by Enum")
                return True
        except Exception:
            pass
        print(f"[HIK][direct][WARN] {name}={value} not supported or set failed")
        return False

    def _set_enum_string_if_possible(self, name: str, value: str, required: bool = False) -> bool:
        try:
            fn = getattr(self.cam, "MV_CC_SetEnumValueByString", None)
            if fn is not None:
                ret = fn(name, str(value))
                if ret == 0:
                    print(f"[HIK][direct][sync] set {name}={value}")
                    return True
                msg = f"ret=0x{ret:x}"
            else:
                msg = "MV_CC_SetEnumValueByString not found"
            # Fallback only for a few common enum nodes.
            fallback = None
            key = f"{name}:{value}".lower()
            if key == "triggermode:on":
                fallback = getattr(self.mvs, "MV_TRIGGER_MODE_ON", 1)
            elif key == "triggermode:off":
                fallback = getattr(self.mvs, "MV_TRIGGER_MODE_OFF", 0)
            elif key == "triggersource:software":
                fallback = getattr(self.mvs, "MV_TRIGGER_SOURCE_SOFTWARE", 7)
            if fallback is not None:
                ret = self.cam.MV_CC_SetEnumValue(name, int(fallback))
                if ret == 0:
                    print(f"[HIK][direct][sync] set {name}={value} by numeric fallback {fallback}")
                    return True
                msg = f"{msg}; numeric fallback ret=0x{ret:x}"
            if required:
                raise RuntimeError(f"set enum {name}={value} failed: {msg}")
            print(f"[HIK][direct][sync][WARN] set {name}={value} failed: {msg}")
            return False
        except Exception as e:
            if required:
                raise
            print(f"[HIK][direct][sync][WARN] set {name}={value} exception: {type(e).__name__}: {e}")
            return False

    def _set_bool_if_possible(self, name: str, value: bool) -> bool:
        try:
            ret = self.cam.MV_CC_SetBoolValue(name, bool(value))
            if ret == 0:
                print(f"[HIK][direct][sync] set {name}={bool(value)}")
                return True
            print(f"[HIK][direct][sync][WARN] set {name}={bool(value)} failed ret=0x{ret:x}")
        except Exception as e:
            print(f"[HIK][direct][sync][WARN] set {name}={bool(value)} exception: {type(e).__name__}: {e}")
        return False

    def _set_float_if_possible(self, name: str, value: Optional[float]) -> bool:
        if value is None:
            return False
        try:
            ret = self.cam.MV_CC_SetFloatValue(name, float(value))
            if ret == 0:
                print(f"[HIK][direct][sync] set {name}={float(value)}")
                return True
            print(f"[HIK][direct][sync][WARN] set {name}={float(value)} failed ret=0x{ret:x}")
        except Exception as e:
            print(f"[HIK][direct][sync][WARN] set {name}={float(value)} exception: {type(e).__name__}: {e}")
        return False

    @staticmethod
    def _float_param_value(param: Any, names: List[str]) -> Optional[float]:
        for n in names:
            if hasattr(param, n):
                try:
                    return float(getattr(param, n))
                except Exception:
                    pass
        return None

    @staticmethod
    def _fmt_float_param(value: Optional[float]) -> str:
        return "NA" if value is None else f"{float(value):.6g}"

    def _get_float_if_possible(self, name: str) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        fn = getattr(self.cam, "MV_CC_GetFloatValue", None)
        if fn is None:
            print(f"[HIK][direct][sync][WARN] readback {name} failed: MV_CC_GetFloatValue not found", flush=True)
            return None

        last_ret: Optional[int] = None
        last_exc: Optional[Exception] = None
        tried_struct = False
        for cls_name in ("MVCC_FLOATVALUE", "MVCC_FLOATVALUE_EX", "MVCC_FLOATVALUE_T"):
            cls = getattr(self.mvs, cls_name, None)
            if cls is None:
                continue
            tried_struct = True
            try:
                param = cls()
                ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
                ret = None
                call_exc: Optional[Exception] = None
                for arg in (param, ctypes.byref(param)):
                    try:
                        ret = fn(name, arg)
                        break
                    except Exception as e:
                        call_exc = e
                if ret is None:
                    if call_exc is not None:
                        raise call_exc
                    continue
                last_ret = int(ret)
                if ret != 0:
                    continue
                cur = self._float_param_value(param, ["fCurValue", "fCur", "fValue", "nCurValue", "nCur"])
                min_v = self._float_param_value(param, ["fMin", "fMinValue", "nMin", "nMinValue"])
                max_v = self._float_param_value(param, ["fMax", "fMaxValue", "nMax", "nMaxValue"])
                print(
                    f"[HIK][direct][sync] readback {name}: "
                    f"cur={self._fmt_float_param(cur)} "
                    f"min={self._fmt_float_param(min_v)} "
                    f"max={self._fmt_float_param(max_v)}",
                    flush=True,
                )
                return cur, min_v, max_v
            except Exception as e:
                last_exc = e
                continue

        if last_ret is not None:
            print(f"[HIK][direct][sync][WARN] readback {name} failed ret=0x{last_ret:x}", flush=True)
        elif last_exc is not None:
            print(f"[HIK][direct][sync][WARN] readback {name} exception: {type(last_exc).__name__}: {last_exc}", flush=True)
        elif not tried_struct:
            print(f"[HIK][direct][sync][WARN] readback {name} failed: no MVCC_FLOATVALUE struct in SDK wrapper", flush=True)
        return None

    def _configure_sync_features_before_start(self) -> None:
        cfg = self.sync_cfg if isinstance(self.sync_cfg, dict) else {}
        if not _sync_enabled(cfg):
            return
        # 固定曝光 / 增益，避免人体帧曝光窗口不稳定。
        if _as_bool(cfg.get("exposure_auto_off", True), True):
            self._set_enum_string_if_possible("ExposureAuto", "Off", required=False)
        if cfg.get("exposure_us", None) is not None:
            self._set_float_if_possible("ExposureTime", float(cfg.get("exposure_us")))
            self._get_float_if_possible("ExposureTime")
        if _as_bool(cfg.get("gain_auto_off", True), True):
            self._set_enum_string_if_possible("GainAuto", "Off", required=False)
        if cfg.get("gain", None) is not None:
            self._set_float_if_possible("Gain", float(cfg.get("gain")))
            self._get_float_if_possible("Gain")

        # 第一阶段联动版默认仍使用连续采集：MVS 每帧曝光开始输出 Line1 脉冲。
        # 如果这里改成 TriggerMode=On，必须另有统一 soft-trigger 控制器，否则 GetImageBuffer 会等不到图像。
        trigger_mode = str(cfg.get("mvs_trigger_mode", "Off") or "Off")
        self._set_enum_string_if_possible("AcquisitionMode", "Continuous", required=False)
        self._set_enum_string_if_possible("TriggerMode", trigger_mode, required=False)
        if trigger_mode.strip().lower() == "on":
            self._set_enum_string_if_possible("TriggerSource", str(cfg.get("trigger_source", "Software") or "Software"), required=False)
            if _soft_trigger_enabled(cfg):
                print("[HIK][direct][sync] TriggerMode=On; central 40ms software trigger loop will drive frames.", flush=True)
            else:
                print("[HIK][direct][sync][WARN] TriggerMode=On needs an external/central software trigger loop; direct human module does not trigger frames.", flush=True)

        if _as_bool(cfg.get("enable_mvs_line_output", True), True):
            print("[HIK][direct][sync] configuring Line output: ExposureStartActive pulse", flush=True)
            self._set_enum_string_if_possible("LineSelector", str(cfg.get("line_selector", "Line1") or "Line1"), required=False)
            self._set_enum_string_if_possible("LineMode", str(cfg.get("line_mode", "Strobe") or "Strobe"), required=False)
            self._set_enum_string_if_possible("LineSource", str(cfg.get("line_source", "ExposureStartActive") or "ExposureStartActive"), required=False)
            if "line_inverter" in cfg:
                self._set_bool_if_possible("LineInverter", _as_bool(cfg.get("line_inverter", False), False))
            self._set_bool_if_possible(str(cfg.get("strobe_enable_feature", "StrobeEnable") or "StrobeEnable"), True)
            if cfg.get("strobe_duration_us", None) is not None:
                self._set_float_if_possible("StrobeLineDuration", float(cfg.get("strobe_duration_us")))
            if cfg.get("strobe_delay_us", None) is not None:
                self._set_float_if_possible("StrobeLineDelay", float(cfg.get("strobe_delay_us")))
            if cfg.get("strobe_pre_delay_us", None) is not None:
                self._set_float_if_possible("StrobeLinePreDelay", float(cfg.get("strobe_pre_delay_us")))
            print("[HIK][direct][sync] Line output config done", flush=True)

    def _reset_native_sampling(self) -> None:
        """同型号相机统一原生画幅。

        旧版本为了兼容一台高分辨率相机，允许在 JSON 中配置 Decimation/Binning。
        现在所有可见光相机都换成同一型号后，不再使用降采样。
        但海康相机的 Decimation/Binning 可能会在设备节点或 UserSet 中残留，
        所以这里启动时强制复位为 1，保证两台同型号相机输出同样的原生完整画幅。
        """
        for name in ["DecimationHorizontal", "DecimationVertical", "BinningHorizontal", "BinningVertical"]:
            self._set_int_or_enum(name, 1)

    def _get_int_current(self, name: str) -> Optional[int]:
        p = self._get_int_param(name)
        if p is None:
            return None
        return self._param_value(p, ["nCurValue", "nCur", "nValue"], None)

    def _print_roi_status(self, stage: str) -> None:
        vals = []
        for name in ["Width", "Height", "OffsetX", "OffsetY", "DecimationHorizontal", "DecimationVertical", "BinningHorizontal", "BinningVertical"]:
            v = self._get_int_current(name)
            if v is not None:
                vals.append(f"{name}={v}")
        if vals:
            print(f"[HIK][direct][roi][{stage}] " + ", ".join(vals))

    def _set_int_min(self, name: str) -> None:
        p = self._get_int_param(name)
        if p is None:
            self._set_int_silent(name, 0, print_ok=True)
            return
        v = self._param_value(p, ["nMin", "nMinValue"], 0)
        if v is not None:
            self._set_int_silent(name, int(v), print_ok=True)

    def _set_int_max(self, name: str) -> None:
        p = self._get_int_param(name)
        if p is None:
            print(f"[HIK][direct][WARN] 读取 {name}.Max 失败，跳过")
            return
        v = self._param_value(p, ["nMax", "nMaxValue"], None)
        if v is not None and int(v) > 0:
            self._set_int_silent(name, int(v), print_ok=True)

    def _restore_full_roi(self) -> None:
        print("[HIK][direct] restore full sensor ROI: OffsetX/OffsetY=min, Width/Height=max")
        # 先把偏移置零，否则 Width/Height 可能无法设置到最大。
        self._set_int_min("OffsetX")
        self._set_int_min("OffsetY")
        # 再把 ROI 尺寸设到最大，保证不是只取上半幅。
        self._set_int_max("Width")
        self._set_int_max("Height")
        # 部分型号设置 Width/Height 后 Offset 会受影响，再置零一次。
        self._set_int_min("OffsetX")
        self._set_int_min("OffsetY")


    def _force_set_bgr8_pixel_format(self) -> bool:
        """在 StartGrabbing 前强制请求相机端输出 BGR8_Packed。

        这里不改同步/触发/曝光逻辑，只设置 PixelFormat。
        设置成功不等于每帧一定已经是 BGR8，因此 _convert_to_bgr() 仍会用
        stFrameInfo.enPixelType 做最终判断：BGR8 走 fast path，非 BGR8 走旧转换兜底。
        """
        pix = getattr(self.mvs, "PixelType_Gvsp_BGR8_Packed", None)
        if pix is None:
            print("[HIK][direct][BGR8_FORCE][WARN] SDK wrapper has no PixelType_Gvsp_BGR8_Packed", flush=True)
            return False
        try:
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", int(pix))
            if ret == 0:
                print("[HIK][direct][BGR8_FORCE] set PixelFormat=PixelType_Gvsp_BGR8_Packed before StartGrabbing", flush=True)
                return True
            print(f"[HIK][direct][BGR8_FORCE][WARN] set PixelFormat=BGR8_Packed failed ret=0x{ret:x}; will verify frame pixel type and fallback if needed", flush=True)
        except Exception as e:
            print(f"[HIK][direct][BGR8_FORCE][WARN] set PixelFormat=BGR8_Packed exception: {type(e).__name__}: {e}; will fallback if needed", flush=True)
        return False

    def _configure_before_start(self) -> None:
        mvs = self.mvs
        cam = self.cam
        try:
            cam.MV_CC_SetEnumValue("TriggerMode", getattr(mvs, "MV_TRIGGER_MODE_OFF", 0))
        except Exception:
            pass

        # 同型号相机统一原生画幅：先清除设备中可能残留的 Decimation/Binning，再恢复完整 ROI。
        self._reset_native_sampling()
        self._print_roi_status("after_sampling_reset")

        if self.force_full_sensor and self.width <= 0 and self.height <= 0:
            self._restore_full_roi()
            self._print_roi_status("after_full_roi")
        else:
            # 注意：这不是缩放，而是相机端 ROI 裁剪。只有明确想裁剪时才设置。
            if self.width > 0:
                self._set_int_silent("Width", self.width, print_ok=True)
            if self.height > 0:
                self._set_int_silent("Height", self.height, print_ok=True)

        if self.fps > 0:
            try:
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.fps))
                print(f"[HIK][direct] set AcquisitionFrameRate={float(self.fps):.2f}")
            except Exception:
                print("[HIK][direct][WARN] 设置 AcquisitionFrameRate 失败，忽略")

        self._configure_sync_features_before_start()

        # 安全版：只在 JSON 显式 prefer_bgr8=true 时请求 BGR8_Packed。
        # 注意：这里只设置采集格式，不改变同步/触发/子弹轨迹逻辑。
        if self.prefer_bgr8:
            self._force_set_bgr8_pixel_format()

    def _start_grabbing(self) -> None:
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:x}")

    def software_trigger(self) -> Tuple[bool, int, int, int]:
        """发送一次 MVS 软件触发命令。

        返回: (ok, cmd_wall_us, done_wall_us, ret)
        注意：真正给事件相机的硬件同步点仍然是 ExposureStartActive，
        这里的 cmd_wall_us 只是排查软件命令抖动用。
        """
        if self.cam is None:
            return False, now_us(), now_us(), -1
        t0 = now_us()
        try:
            ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
        except Exception as e:
            t1 = now_us()
            print(f"[HIK][direct][soft_trigger][{self.serial}][WARN] TriggerSoftware exception: {type(e).__name__}: {e}", flush=True)
            return False, t0, t1, -2
        t1 = now_us()
        if ret != 0:
            print(f"[HIK][direct][soft_trigger][{self.serial}][WARN] TriggerSoftware failed ret=0x{ret:x}", flush=True)
        return ret == 0, t0, t1, int(ret)

    def _convert_to_bgr(self, frame_out: Any) -> np.ndarray:
        """Return an OpenCV/TensorRT-ready BGR image.

        BGR8 fast path:
        - If the Hikrobot camera actually outputs PixelType_Gvsp_BGR8_Packed,
          the SDK buffer is already HxWx3 BGR.
        - In that case this function only wraps + copies the SDK buffer and
          intentionally skips MV_CC_ConvertPixelType and cv2.cvtColor.
        - The copy is kept because MV_CC_FreeImageBuffer() releases/reuses the
          SDK buffer immediately after read(), while later YOLO/SHM/UE code may
          still hold the numpy image.

        Fallback:
        - Non-BGR8 camera output keeps the old SDK conversion path, so sync,
          bullet trajectory detection, event trigger, JSONL, SHM and UE output
          logic are not changed by this patch.
        """
        mvs = self.mvs
        info = frame_out.stFrameInfo
        w = int(info.nWidth)
        h = int(info.nHeight)
        src_len = int(info.nFrameLen)
        src_pixel_type = int(info.enPixelType)
        expected_bgr_len = w * h * 3

        raw = np.ctypeslib.as_array((ctypes.c_ubyte * src_len).from_address(frame_out.pBufAddr))

        bgr_pixel = getattr(mvs, "PixelType_Gvsp_BGR8_Packed", None)
        if bgr_pixel is not None and src_pixel_type == int(bgr_pixel):
            if src_len < expected_bgr_len:
                raise RuntimeError(
                    f"BGR8 frame length too small: len={src_len}, expected={expected_bgr_len}, "
                    f"pixel_type={src_pixel_type}"
                )
            if not getattr(self, "_bgr8_fast_path_printed", False):
                self._bgr8_fast_path_printed = True
                cam_tag = getattr(self, "serial", "") or getattr(getattr(self, "cfg", None), "camera_id", "") or "unknown"
                print(
                    f"[HIK][BGR8_FAST][{cam_tag}] direct BGR8_Packed path enabled: "
                    f"{w}x{h}, len={src_len}, pixel_type={src_pixel_type}; "
                    f"skip MV_CC_ConvertPixelType/cvtColor",
                    flush=True,
                )
            return raw[:expected_bgr_len].reshape(h, w, 3).copy()

        rgb_pixel = getattr(mvs, "PixelType_Gvsp_RGB8_Packed", None)
        if rgb_pixel is not None and src_pixel_type == int(rgb_pixel) and src_len >= expected_bgr_len:
            if not getattr(self, "_rgb8_path_printed", False):
                self._rgb8_path_printed = True
                cam_tag = getattr(self, "serial", "") or getattr(getattr(self, "cfg", None), "camera_id", "") or "unknown"
                print(
                    f"[HIK][RGB8_WARN][{cam_tag}] camera returned RGB8_Packed, not BGR8_Packed: "
                    f"{w}x{h}, len={src_len}, pixel_type={src_pixel_type}; "
                    f"using cvtColor RGB->BGR fallback",
                    flush=True,
                )
            img = raw[:expected_bgr_len].reshape(h, w, 3).copy()
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dst_pixel = bgr_pixel
        need_rgb_to_bgr = False
        if dst_pixel is None:
            dst_pixel = getattr(mvs, "PixelType_Gvsp_RGB8_Packed")
            need_rgb_to_bgr = True

        dst_size = expected_bgr_len
        dst_buf = (ctypes.c_ubyte * dst_size)()
        param = mvs.MV_CC_PIXEL_CONVERT_PARAM()
        ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.pSrcData = frame_out.pBufAddr
        param.nSrcDataLen = src_len
        param.enSrcPixelType = src_pixel_type
        param.enDstPixelType = dst_pixel
        param.pDstBuffer = dst_buf
        param.nDstBufferSize = dst_size

        if not getattr(self, "_sdk_convert_path_printed", False):
            self._sdk_convert_path_printed = True
            cam_tag = getattr(self, "serial", "") or getattr(getattr(self, "cfg", None), "camera_id", "") or "unknown"
            print(
                f"[HIK][SDK_CONVERT][{cam_tag}] source pixel_type={src_pixel_type}, len={src_len}; "
                f"using MV_CC_ConvertPixelType -> BGR8 fallback",
                flush=True,
            )
        ret = self.cam.MV_CC_ConvertPixelType(param)
        if ret != 0:
            if src_len == w * h:
                gray = raw.reshape(h, w).copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if src_len >= expected_bgr_len:
                return raw[:expected_bgr_len].reshape(h, w, 3).copy()
            raise RuntimeError(f"MV_CC_ConvertPixelType failed: 0x{ret:x}, pixel_type={src_pixel_type}, len={src_len}")
        img = np.ctypeslib.as_array(dst_buf).reshape(h, w, 3).copy()
        if need_rgb_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        mvs = self.mvs
        frame_out = mvs.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
        ret = self.cam.MV_CC_GetImageBuffer(frame_out, self.timeout_ms)
        if ret != 0:
            return False, None, now_us()
        try:
            info = frame_out.stFrameInfo
            frame_num = int(getattr(info, "nFrameNum", 0) or 0)
            self.last_mvs_frame_num = frame_num
            dev_ts_high = int(getattr(info, "nDevTimeStampHigh", 0) or 0)
            dev_ts_low = int(getattr(info, "nDevTimeStampLow", 0) or 0)
            self.last_mvs_frame_info = {
                "nFrameNum": frame_num,
                "nWidth": int(getattr(info, "nWidth", 0) or 0),
                "nHeight": int(getattr(info, "nHeight", 0) or 0),
                "nFrameLen": int(getattr(info, "nFrameLen", 0) or 0),
                "enPixelType": int(getattr(info, "enPixelType", 0) or 0),
                "nDevTimeStampHigh": dev_ts_high,
                "nDevTimeStampLow": dev_ts_low,
                "dev_timestamp_64": int((dev_ts_high << 32) | dev_ts_low) if (dev_ts_high or dev_ts_low) else 0,
                "host_read_wall_us": now_us(),
            }
            frame = self._convert_to_bgr(frame_out)
            if not self._first_shape_printed:
                self._first_shape_printed = True
                print(f"[HIK][direct] first_frame_shape={frame.shape} mvs_frame_num={frame_num}  # H,W,C；同型号相机应一致")
            return True, frame, now_us()
        finally:
            try:
                self.cam.MV_CC_FreeImageBuffer(frame_out)
            except Exception:
                pass

    def release(self) -> None:
        if self.cam is None:
            return
        for fn in ["MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"]:
            try:
                getattr(self.cam, fn)()
            except Exception:
                pass


class DirectFrameWorker:
    def __init__(self, cam_id: str, args: argparse.Namespace):
        self.cam_id = str(cam_id)
        self.args = args
        self.lock = threading.Lock()
        self.latest: Optional[Tuple[np.ndarray, int, int, int, Dict[str, Any], float]] = None
        self.frame_count = 0
        self.fps_value = 0.0
        self.last_error = ""
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.source: Optional[BaseFrameSource] = None
        self.ended = False
        self.mvs_shm_publisher = MvsSharedFramePublisher(getattr(args, "sync_cfg", {}) or {}, [self.cam_id])
        self.mvs_frame_audit = MvsFrameAuditWriter(getattr(args, "sync_cfg", {}) or {}, [self.cam_id])

    def _open_source(self) -> BaseFrameSource:
        if self.args.source == "hik":
            return HikFullViewSource(
                serial=self.args.serial,
                mvs_sdk_path=self.args.mvs_sdk_path,
                timeout_ms=self.args.hik_timeout_ms,
                width=self.args.capture_width,
                height=self.args.capture_height,
                fps=self.args.capture_fps,
                force_full_sensor=bool(getattr(self.args, "force_full_sensor", True)),
                prefer_bgr8=bool(getattr(self.args, "prefer_bgr8", False)),
                sync_cfg=getattr(self.args, "sync_cfg", {}) or {},
            )
        if self.args.source == "video":
            return VideoFileSource(self.args.video, backend=self.args.video_backend)
        raise ValueError(f"未知 source: {self.args.source}")

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._run, name=f"direct_capture_{self.cam_id}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.source is not None:
            self.source.release()
            self.source = None
        try:
            self.mvs_shm_publisher.close()
        except Exception:
            pass
        try:
            self.mvs_frame_audit.close()
        except Exception:
            pass

    def _run(self) -> None:
        t0 = time.time()
        n = 0
        last_video_tick = None
        try:
            self.source = self._open_source()
            print(f"[capture][{self.cam_id}] opened: {self.source.source_name}, source={self.args.source}")
            while self.running:
                ok, frame, ts_us = self.source.read()
                if not ok or frame is None:
                    if self.args.source == "video":
                        if bool(getattr(self.args, "video_loop", False)):
                            try:
                                self.source.release()
                            except Exception:
                                pass
                            self.source = self._open_source()
                            last_video_tick = None
                            continue
                        self.ended = True
                        self.running = False
                        print(f"[capture][{self.cam_id}] video end, loop=false, stop worker")
                        break
                    time.sleep(0.005)
                    continue

                # 这里保留 resize 参数，但默认配置为 0。要保证完整画面，就不要在这里裁剪。
                # cv2.resize 是完整缩放，不会裁剪；但是默认不启用，显示缩小由窗口显示函数完成。
                if int(getattr(self.args, "resize_width", 0)) > 0 and int(getattr(self.args, "resize_height", 0)) > 0:
                    frame = cv2.resize(frame, (int(self.args.resize_width), int(self.args.resize_height)), interpolation=cv2.INTER_AREA)

                self.frame_count += 1
                n += 1
                mvs_frame_num = int(getattr(self.source, "last_mvs_frame_num", 0) or self.frame_count)
                mvs_meta = copy.deepcopy(getattr(self.source, "last_mvs_frame_info", {}) or {})
                with self.lock:
                    self.latest = (frame, int(ts_us), int(self.frame_count), int(mvs_frame_num), mvs_meta, time.monotonic())
                self.mvs_shm_publisher.write(self.cam_id, frame, int(ts_us), int(self.frame_count), int(mvs_frame_num), mvs_meta)
                self.mvs_frame_audit.write(self.cam_id, frame, int(ts_us), int(self.frame_count), int(mvs_frame_num), mvs_meta)

                if self.args.source == "video" and bool(getattr(self.args, "video_realtime", True)):
                    fps = float(getattr(self.source, "fps", 25.0) or 25.0)
                    if fps > 1e-6:
                        target_dt = 1.0 / fps
                        if last_video_tick is not None:
                            elapsed = time.monotonic() - last_video_tick
                            if elapsed < target_dt:
                                time.sleep(target_dt - elapsed)
                        last_video_tick = time.monotonic()

                if time.time() - t0 >= 1.0:
                    self.fps_value = n / max(1e-6, time.time() - t0)
                    t0 = time.time()
                    n = 0
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[capture][{self.cam_id}][ERROR] {self.last_error}")
            traceback.print_exc()
        finally:
            if self.source is not None:
                self.source.release()
                self.source = None

    def get_latest(self) -> Optional[Tuple[np.ndarray, int, int, int, Dict[str, Any], float]]:
        with self.lock:
            if self.latest is None:
                return None
            frame, ts_us, fid, mvs_frame_num, mvs_meta, t = self.latest
            return frame.copy(), ts_us, fid, mvs_frame_num, copy.deepcopy(mvs_meta), t



# =============================================================================
# One-key raw MVS recording
# =============================================================================

def _record_cfg_enabled(cfg: Dict[str, Any]) -> bool:
    return _as_bool((cfg or {}).get("enable", False), False)


def _record_command_file(record_cfg: Dict[str, Any], sync_cfg: Dict[str, Any]) -> str:
    root = str((sync_cfg or {}).get("root", "/home/ysxq/PycharmProjects/Ids_Test_3.9/test607/sync_ipc"))
    default = os.path.join(root, "record_control.json")
    return str((record_cfg or {}).get("command_file", default))


def _record_root(record_cfg: Dict[str, Any]) -> str:
    return str((record_cfg or {}).get("record_root", "/home/ysxq/PycharmProjects/Ids_Test_3.9/test607/recordings"))


def _safe_session_id(session_id: Optional[str] = None) -> str:
    raw = str(session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:96] or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_one_key_record_command(command_file: str, cmd: str, session_id: Optional[str] = None) -> bool:
    payload: Dict[str, Any] = {
        "seq": int(time.time() * 1000),
        "cmd": str(cmd).lower(),
        "wall_us": now_us(),
        "source": "rentijiance_mvs_window",
    }
    if session_id:
        payload["session_id"] = str(session_id)
    try:
        path = Path(command_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(f"[onekey_record] command {cmd} -> {path} session={payload.get('session_id','')}", flush=True)
        return True
    except Exception as e:
        print(f"[onekey_record][ERROR] write command failed: {command_file}: {e}", flush=True)
        return False


class MvsRawRecorderManager:
    """Non-blocking raw MVS AVI recorder controlled by sync_ipc/record_control.json.

    It records frames before YOLO/draw overlay. The receiver thread only enqueues a copy; actual AVI/CSV writing
    is done in writer threads so 40 ms software trigger timing is not blocked.
    """

    def __init__(self,
                 record_cfg: Dict[str, Any],
                 sync_cfg: Dict[str, Any],
                 route_order: List[str],
                 route_args: Dict[str, argparse.Namespace]):
        self.cfg = record_cfg if isinstance(record_cfg, dict) else {}
        self.sync_cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.route_order = list(route_order)
        self.route_args = route_args
        self.enabled = _record_cfg_enabled(self.cfg)
        self.command_file = _record_command_file(self.cfg, self.sync_cfg)
        self.status_file = str(self.cfg.get("status_file", os.path.join(str(self.sync_cfg.get("root", "./sync_ipc") or "./sync_ipc"), "record_status.json")))
        self.record_root = _record_root(self.cfg)
        self.fourcc = str(self.cfg.get("fourcc", "XVID") or "XVID")[:4]
        self.video_fps = float(self.cfg.get("video_fps", 25.0) or 25.0)
        self.queue_size = max(8, int(self.cfg.get("queue_size", 128) or 128))
        self.poll_interval_s = max(0.01, float(self.cfg.get("record_control_poll_ms", 20) or 20) / 1000.0)
        self.record_raw_mvs_avi = _as_bool(self.cfg.get("record_raw_mvs_avi", True), True)
        self.prewarm_sec = float(self.cfg.get("prewarm_sec", 30) or 30)
        self.discard_tail_sec = float(self.cfg.get("discard_tail_sec", 30) or 30)
        self.include_frame_index = _as_bool(self.cfg.get("write_frame_index_csv", True), True)
        self._lock = threading.Lock()
        self._recording = False
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._last_seq: Any = None
        self._closed = False
        self._poll_thread: Optional[threading.Thread] = None
        self._writer_threads: Dict[str, threading.Thread] = {}
        self._queues: Dict[str, queue.Queue] = {}
        self._dropped: Dict[str, int] = {cid: 0 for cid in self.route_order}
        self._written: Dict[str, int] = {cid: 0 for cid in self.route_order}
        self._writer_state: Dict[str, Dict[str, Any]] = {cid: {} for cid in self.route_order}
        for cid in self.route_order:
            self._queues[cid] = queue.Queue(maxsize=self.queue_size)

    def route_enabled(self, cam_id: str) -> bool:
        if not self.enabled:
            return False
        args = self.route_args.get(cam_id)
        if args is not None and hasattr(args, "one_key_record_enable"):
            return bool(getattr(args, "one_key_record_enable"))
        return True

    def start(self) -> None:
        if not self.enabled:
            return
        Path(self.record_root).mkdir(parents=True, exist_ok=True)
        Path(self.command_file).parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        for cid in self.route_order:
            th = threading.Thread(target=self._writer_loop, args=(cid,), name=f"mvs_raw_record_writer_{cid}", daemon=True)
            self._writer_threads[cid] = th
            th.start()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="mvs_raw_record_control_poll", daemon=True)
        self._poll_thread.start()
        enabled_routes = [cid for cid in self.route_order if self.route_enabled(cid)]
        print(
            f"[onekey_record] MVS raw recorder enabled. command_file={self.command_file} root={self.record_root} routes={enabled_routes}",
            flush=True,
        )
        print("[onekey_record] Press S in MVS/event/OpenGL window or launcher terminal to start/stop all enabled recorders.", flush=True)
        self._write_status(recording=False, session_id=None, reason="ready")

    def close(self) -> None:
        self._closed = True
        if self.enabled:
            self.stop_recording(reason="close")
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.5)
        for cid, q in self._queues.items():
            try:
                q.put_nowait({"_cmd": "shutdown"})
            except Exception:
                pass
        for th in self._writer_threads.values():
            th.join(timeout=2.0)

    def is_recording(self) -> bool:
        with self._lock:
            return bool(self._recording)

    def current_session_id(self) -> Optional[str]:
        with self._lock:
            return self._session_id

    def _write_status(self, recording: bool, session_id: Optional[str], reason: str = "") -> None:
        try:
            path = Path(self.status_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "recording": bool(recording),
                "session_id": str(session_id or ""),
                "reason": str(reason or ""),
                "wall_us": now_us(),
                "record_root": self.record_root,
                "command_file": self.command_file,
                "status_file": self.status_file,
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            print(f"[onekey_record][WARN] write status failed: {e}", flush=True)

    def start_recording(self, session_id: Optional[str] = None, source: str = "command") -> None:
        if not self.enabled:
            return
        sid = _safe_session_id(session_id)
        with self._lock:
            if self._recording and self._session_id == sid:
                return
            if self._recording:
                # close old session first
                old_sid = self._session_id
                print(f"[onekey_record] switching MVS recording session {old_sid} -> {sid}", flush=True)
                self._enqueue_close_session()
            self._recording = True
            self._session_id = sid
            self._session_dir = Path(self.record_root) / sid
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest_start(source=source)
        self._write_status(recording=True, session_id=sid, reason=source)
        print(f"[onekey_record] MVS recording START session={sid} dir={self._session_dir}", flush=True)

    def stop_recording(self, reason: str = "command") -> None:
        with self._lock:
            if not self._recording:
                return
            sid = self._session_id
            session_dir = self._session_dir
            self._recording = False
            self._session_id = None
            self._session_dir = None
            self._write_manifest_stop(session_dir, sid, reason=reason)
            self._enqueue_close_session()
        self._write_status(recording=False, session_id=sid, reason=reason)
        print(f"[onekey_record] MVS recording STOP session={sid} reason={reason}", flush=True)

    def toggle_recording(self) -> None:
        if self.is_recording():
            self.stop_recording(reason="toggle")
        else:
            self.start_recording(session_id=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), source="toggle")

    def _enqueue_close_session(self) -> None:
        for q in self._queues.values():
            try:
                q.put_nowait({"_cmd": "close_session"})
            except Exception:
                pass

    def _write_manifest_start(self, source: str) -> None:
        if self._session_dir is None:
            return
        enabled_routes = []
        for cid in self.route_order:
            args = self.route_args.get(cid)
            enabled_routes.append({
                "camera": cid,
                "enabled": bool(self.route_enabled(cid)),
                "serial": str(getattr(args, "serial", "")) if args is not None else "",
                "source": str(getattr(args, "source", "")) if args is not None else "",
            })
        manifest = {
            "session_id": self._session_id,
            "start_wall_us": now_us(),
            "record_root": self.record_root,
            "command_file": self.command_file,
            "source": source,
            "type": "raw_mvs_bgr_avi_no_detection_overlay",
            "video_fps": self.video_fps,
            "fourcc": self.fourcc,
            "prewarm_sec": self.prewarm_sec,
            "discard_tail_sec": self.discard_tail_sec,
            "routes": enabled_routes,
            "soft_trigger_period_ms": float(self.sync_cfg.get("mvs_soft_trigger_period_ms", 40.0) or 40.0),
            "mvs_trigger_mode": str(self.sync_cfg.get("mvs_trigger_mode", "")),
            "line_source": str(self.sync_cfg.get("line_source", "")),
        }
        try:
            (self._session_dir / "manifest_mvs.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[onekey_record][WARN] write manifest failed: {e}", flush=True)

    def _write_manifest_stop(self, session_dir: Optional[Path], sid: Optional[str], reason: str) -> None:
        if session_dir is None:
            return
        path = session_dir / "manifest_mvs_stop.json"
        data = {
            "session_id": sid,
            "stop_wall_us": now_us(),
            "reason": reason,
            "written_frames": dict(self._written),
            "dropped_frames": dict(self._dropped),
        }
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[onekey_record][WARN] write stop manifest failed: {e}", flush=True)

    def _poll_loop(self) -> None:
        while not self._closed:
            self._poll_once()
            time.sleep(self.poll_interval_s)

    def _poll_once(self) -> None:
        path = Path(self.command_file)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        seq = data.get("seq")
        if seq is None:
            try:
                seq = int(path.stat().st_mtime_ns)
            except Exception:
                seq = None
        if seq == self._last_seq:
            return
        self._last_seq = seq
        cmd = str(data.get("cmd", data.get("command", ""))).lower().strip()
        sid = data.get("session_id")
        if cmd == "start":
            self.start_recording(session_id=sid, source=str(data.get("source", "command")))
        elif cmd == "stop":
            self.stop_recording(reason=str(data.get("source", "command")))
        elif cmd == "toggle":
            self.toggle_recording()

    def write_frame(self, cam_id: str, frame: np.ndarray, ts_us: int, fid: int, mvs_frame_num: int, mvs_meta: Dict[str, Any]) -> None:
        if not self.enabled or not self.route_enabled(cam_id):
            return
        with self._lock:
            if not self._recording or self._session_id is None or self._session_dir is None:
                return
            sid = self._session_id
            session_dir = self._session_dir
        if frame is None:
            return
        item = {
            "session_id": sid,
            "session_dir": str(session_dir),
            "cam_id": str(cam_id),
            "serial": str(getattr(self.route_args.get(cam_id), "serial", "")),
            "frame": frame.copy(),
            "ts_us": int(ts_us),
            "fid": int(fid),
            "mvs_frame_num": int(mvs_frame_num),
            "mvs_meta": copy.deepcopy(mvs_meta or {}),
            "record_wall_us": int(now_us()),
        }
        q = self._queues.get(cam_id)
        if q is None:
            return
        try:
            q.put_nowait(item)
        except queue.Full:
            self._dropped[cam_id] = int(self._dropped.get(cam_id, 0)) + 1
            if self._dropped[cam_id] <= 20:
                print(f"[onekey_record][{cam_id}][WARN] recording queue full; dropped={self._dropped[cam_id]}", flush=True)

    def _writer_loop(self, cam_id: str) -> None:
        q = self._queues[cam_id]
        writer = None
        csv_fp = None
        current_session = None
        record_frame_idx = 0
        try:
            while True:
                item = q.get()
                if isinstance(item, dict) and item.get("_cmd") == "shutdown":
                    break
                if isinstance(item, dict) and item.get("_cmd") == "close_session":
                    if writer is not None:
                        writer.release(); writer = None
                    if csv_fp is not None:
                        csv_fp.close(); csv_fp = None
                    current_session = None
                    record_frame_idx = 0
                    continue
                if not isinstance(item, dict):
                    continue
                sid = item.get("session_id")
                if sid != current_session:
                    if writer is not None:
                        writer.release(); writer = None
                    if csv_fp is not None:
                        csv_fp.close(); csv_fp = None
                    current_session = sid
                    record_frame_idx = 0
                    base_dir = Path(str(item.get("session_dir"))) / f"MVS_{cam_id}_{item.get('serial','') or 'unknown'}"
                    base_dir.mkdir(parents=True, exist_ok=True)
                    frame = item["frame"]
                    h, w = frame.shape[:2]
                    if self.record_raw_mvs_avi:
                        fourcc = cv2.VideoWriter_fourcc(*self.fourcc.ljust(4, 'X')[:4])
                        video_path = str(base_dir / "raw_mvs.avi")
                        writer = cv2.VideoWriter(video_path, fourcc, float(self.video_fps), (int(w), int(h)))
                        if not writer.isOpened():
                            print(f"[onekey_record][{cam_id}][ERROR] cannot open VideoWriter: {video_path}", flush=True)
                            writer = None
                    if self.include_frame_index:
                        csv_fp = open(base_dir / "frame_index.csv", "w", encoding="utf-8", buffering=1)
                        csv_fp.write(
                            "record_frame_idx,cam_id,serial,program_sync_index,mvs_frame_num,local_fid,"
                            "capture_ts_us,receiver_wall_us,soft_trigger_tick_wall_us,soft_trigger_cmd_wall_us,"
                            "soft_trigger_cmd_spread_us,record_wall_us,queue_dropped_before\n"
                        )
                    print(f"[onekey_record][{cam_id}] writer opened session={sid} dir={base_dir}", flush=True)
                frame = item["frame"]
                meta = item.get("mvs_meta") or {}
                if writer is not None:
                    writer.write(frame)
                if csv_fp is not None:
                    csv_fp.write(
                        f"{record_frame_idx},{cam_id},{item.get('serial','')},"
                        f"{meta.get('program_sync_index','')},{item.get('mvs_frame_num','')},{item.get('fid','')},"
                        f"{item.get('ts_us','')},{meta.get('receiver_wall_us','')},{meta.get('soft_trigger_tick_wall_us','')},"
                        f"{meta.get('soft_trigger_cmd_wall_us','')},{meta.get('soft_trigger_cmd_spread_us','')},"
                        f"{item.get('record_wall_us','')},{self._dropped.get(cam_id,0)}\n"
                    )
                record_frame_idx += 1
                self._written[cam_id] = int(self._written.get(cam_id, 0)) + 1
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                if csv_fp is not None:
                    csv_fp.close()



class SoftTriggerCaptureGroup:
    """中心 40ms 软件触发 MVS 采集器 v2。

    v1 的问题：一个线程里“触发 A/D -> 顺序等待 A/D 图像 -> 转 BGR”后才进入下一次 40ms，
    实测会把周期拖到 47~50ms，trigger FPS 只有 20~21fps。

    v2 改法：
      1) trigger 线程只负责按绝对时间节拍发送 TriggerSoftware，不等待图像；
      2) 每台 MVS 各有一个 receiver 线程阻塞 GetImageBuffer，收到帧后转 BGR 并更新 latest；
      3) 用每路 pending trigger 队列把 program_sync_index 绑定到收到的 MVS 帧；
      4) YOLO 仍然只读 latest frame，不阻塞 40ms 触发节拍。
    """

    def __init__(self, route_order: List[str], route_args: Dict[str, argparse.Namespace], sync_cfg: Dict[str, Any], mvs_recorder: Optional[MvsRawRecorderManager] = None):
        self.route_order = [cid for cid in route_order if getattr(route_args[cid], "source", "") == "hik"]
        self.route_args = route_args
        self.sync_cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.period_ms = float(self.sync_cfg.get("mvs_soft_trigger_period_ms", 40.0) or 40.0)
        self.period_sec = max(0.001, self.period_ms / 1000.0)
        self.start_index = int(self.sync_cfg.get("mvs_soft_trigger_start_index", 0) or 0)
        self.grab_timeout_ms = int(self.sync_cfg.get("mvs_soft_trigger_grab_timeout_ms", 1000) or 1000)
        self.trigger_parallel = _as_bool(self.sync_cfg.get("mvs_soft_trigger_parallel_cmd", True), True)
        self.running = False
        self.trigger_thread: Optional[threading.Thread] = None
        self.receiver_threads: Dict[str, threading.Thread] = {}
        self.sources: Dict[str, HikFullViewSource] = {}
        self.latest: Dict[str, Tuple[np.ndarray, int, int, int, Dict[str, Any], float]] = {}
        self.frame_count: Dict[str, int] = {cid: 0 for cid in self.route_order}
        self.fps_value: Dict[str, float] = {cid: 0.0 for cid in self.route_order}
        self.last_error: Dict[str, str] = {cid: "" for cid in self.route_order}
        self.lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending_triggers: Dict[str, deque] = {cid: deque(maxlen=256) for cid in self.route_order}
        self._fps_count: Dict[str, int] = {cid: 0 for cid in self.route_order}
        self._fps_t0 = time.time()
        self._trigger_warn_count = 0
        self._grab_warn_count = 0
        self._trigger_count = 0
        self._late_warn_count = 0
        self.mvs_recorder = mvs_recorder
        self.mvs_shm_publisher = MvsSharedFramePublisher(self.sync_cfg, self.route_order)
        self.mvs_frame_audit = MvsFrameAuditWriter(self.sync_cfg, self.route_order)

    def _open_source_for(self, cam_id: str) -> HikFullViewSource:
        args = self.route_args[cam_id]
        args.hik_timeout_ms = int(getattr(args, "hik_timeout_ms", self.grab_timeout_ms) or self.grab_timeout_ms)
        return HikFullViewSource(
            serial=args.serial,
            mvs_sdk_path=args.mvs_sdk_path,
            timeout_ms=args.hik_timeout_ms,
            width=args.capture_width,
            height=args.capture_height,
            fps=args.capture_fps,
            force_full_sensor=bool(getattr(args, "force_full_sensor", True)),
            prefer_bgr8=bool(getattr(args, "prefer_bgr8", False)),
            sync_cfg=getattr(args, "sync_cfg", {}) or {},
        )

    def start(self) -> None:
        if not self.route_order:
            raise RuntimeError("SoftTriggerCaptureGroup requires at least one hik route")
        print(f"[softsync] opening MVS cameras for central software trigger v2: {self.route_order}", flush=True)
        for cid in self.route_order:
            self.sources[cid] = self._open_source_for(cid)
            print(f"[softsync][{cid}] opened: {self.sources[cid].source_name}", flush=True)
        self.running = True
        # 先启动接收线程，等待后续 TriggerSoftware 产生新帧。
        for cid in self.route_order:
            th = threading.Thread(target=self._receiver_loop, args=(cid,), name=f"soft_trigger_receiver_{cid}", daemon=True)
            self.receiver_threads[cid] = th
            th.start()
        self.trigger_thread = threading.Thread(target=self._trigger_loop, name="soft_trigger_40ms_clock", daemon=True)
        self.trigger_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.trigger_thread is not None:
            self.trigger_thread.join(timeout=3.0)
        for th in list(self.receiver_threads.values()):
            th.join(timeout=2.0)
        for src in list(self.sources.values()):
            try:
                src.release()
            except Exception:
                pass
        self.sources.clear()
        self.receiver_threads.clear()
        try:
            self.mvs_shm_publisher.close()
        except Exception:
            pass
        try:
            self.mvs_frame_audit.close()
        except Exception:
            pass

    def _trigger_one(self, cam_id: str, out: Dict[str, Tuple[bool, int, int, int]]) -> None:
        try:
            out[cam_id] = self.sources[cam_id].software_trigger()
        except Exception as e:
            t = now_us()
            out[cam_id] = (False, t, t, -3)
            self.last_error[cam_id] = f"soft_trigger exception: {type(e).__name__}: {e}"

    def _trigger_all(self) -> Dict[str, Tuple[bool, int, int, int]]:
        results: Dict[str, Tuple[bool, int, int, int]] = {}
        if self.trigger_parallel and len(self.route_order) > 1:
            threads = []
            for cid in self.route_order:
                th = threading.Thread(target=self._trigger_one, args=(cid, results), daemon=True)
                threads.append(th)
                th.start()
            for th in threads:
                th.join(timeout=0.2)
            for cid in self.route_order:
                if cid not in results:
                    t = now_us()
                    results[cid] = (False, t, t, -4)
        else:
            for cid in self.route_order:
                self._trigger_one(cid, results)
        return results

    def _trigger_loop(self) -> None:
        sync_index = int(self.start_index)
        base = time.monotonic()
        next_tick = base
        print(f"[softsync] central MVS software trigger v2 started: period_ms={self.period_ms:.3f}, start_index={sync_index}", flush=True)
        while self.running:
            now_m = time.monotonic()
            delay = next_tick - now_m
            if delay > 0:
                time.sleep(min(0.002, delay))
                continue

            tick_wall_us = now_us()
            trigger_results = self._trigger_all()
            done_wall_us = now_us()
            cmd_times = [v[1] for v in trigger_results.values()] + [v[2] for v in trigger_results.values()]
            cmd_spread_us = int(max(cmd_times) - min(cmd_times)) if cmd_times else 0
            meta_base = {
                "program_sync_index": int(sync_index),
                "soft_trigger_tick_wall_us": int(tick_wall_us),
                "soft_trigger_done_wall_us": int(done_wall_us),
                "soft_trigger_cmd_spread_us": int(cmd_spread_us),
                "soft_trigger_period_ms": float(self.period_ms),
                "soft_trigger_mode": "central_software_trigger_v2_decoupled_receiver",
            }
            with self.pending_lock:
                for cid, cmd_tuple in trigger_results.items():
                    ok_cmd, cmd_wall_us, cmd_done_wall_us, cmd_ret = cmd_tuple
                    meta = dict(meta_base)
                    meta.update({
                        "soft_trigger_cmd_wall_us": int(cmd_wall_us),
                        "soft_trigger_cmd_done_wall_us": int(cmd_done_wall_us),
                        "soft_trigger_cmd_ret": int(cmd_ret),
                        "soft_trigger_cmd_ok": bool(ok_cmd),
                    })
                    if ok_cmd:
                        self.pending_triggers[cid].append(meta)
                    else:
                        self.last_error[cid] = f"TriggerSoftware failed ret={cmd_ret} at sync_index={sync_index}"
                        if self._trigger_warn_count < 20:
                            print(f"[softsync][{cid}][WARN] TriggerSoftware failed ret={cmd_ret} sync_index={sync_index}", flush=True)
                            self._trigger_warn_count += 1
            self._trigger_count += 1
            sync_index += 1
            next_tick += self.period_sec

            # 如果系统暂停/调度卡顿严重，不补发历史触发，直接回到当前节拍。
            behind = time.monotonic() - next_tick
            if behind > self.period_sec * 3.0:
                if self._late_warn_count < 10:
                    print(f"[softsync][WARN] trigger clock late by {behind*1000.0:.1f} ms; resync next_tick", flush=True)
                    self._late_warn_count += 1
                next_tick = time.monotonic() + self.period_sec

    def _pop_pending_meta(self, cam_id: str, mvs_frame_num: int) -> Dict[str, Any]:
        with self.pending_lock:
            q = self.pending_triggers.get(cam_id)
            if q is not None and len(q) > 0:
                return dict(q.popleft())
        # 兜底：如果 pending 队列丢失，至少用 MVS frame_num 推断，避免 human_result 缺 sync_index。
        return {
            "program_sync_index": int(mvs_frame_num) - 1 if int(mvs_frame_num) > 0 else None,
            "soft_trigger_mode": "central_software_trigger_v2_fallback_frame_num",
            "soft_trigger_period_ms": float(self.period_ms),
            "soft_trigger_cmd_ok": False,
            "soft_trigger_cmd_ret": -99,
            "soft_trigger_pending_missing": True,
        }

    def _receiver_loop(self, cam_id: str) -> None:
        src = self.sources.get(cam_id)
        if src is None:
            return
        while self.running:
            try:
                ok, frame, ts_us = src.read()
            except Exception as e:
                ok, frame, ts_us = False, None, now_us()
                self.last_error[cam_id] = f"read exception: {type(e).__name__}: {e}"
            if not self.running:
                break
            if not ok or frame is None:
                self.last_error[cam_id] = "GetImageBuffer timeout/fail in receiver"
                if self._grab_warn_count < 20:
                    print(f"[softsync][{cam_id}][WARN] no frame in receiver", flush=True)
                    self._grab_warn_count += 1
                continue

            args = self.route_args[cam_id]
            if int(getattr(args, "resize_width", 0)) > 0 and int(getattr(args, "resize_height", 0)) > 0:
                frame = cv2.resize(frame, (int(args.resize_width), int(args.resize_height)), interpolation=cv2.INTER_AREA)

            self.frame_count[cam_id] = int(self.frame_count.get(cam_id, 0)) + 1
            self._fps_count[cam_id] = int(self._fps_count.get(cam_id, 0)) + 1
            mvs_frame_num = int(getattr(src, "last_mvs_frame_num", 0) or self.frame_count[cam_id])
            mvs_meta = copy.deepcopy(getattr(src, "last_mvs_frame_info", {}) or {})
            mvs_meta.update(self._pop_pending_meta(cam_id, mvs_frame_num))
            mvs_meta.update({
                "receiver_wall_us": int(now_us()),
                "receiver_cam_id": str(cam_id),
            })
            with self.lock:
                self.latest[cam_id] = (frame, int(ts_us), int(self.frame_count[cam_id]), int(mvs_frame_num), mvs_meta, time.monotonic())
            self.mvs_shm_publisher.write(cam_id, frame, int(ts_us), int(self.frame_count[cam_id]), int(mvs_frame_num), mvs_meta)
            self.mvs_frame_audit.write(cam_id, frame, int(ts_us), int(self.frame_count[cam_id]), int(mvs_frame_num), mvs_meta)
            if self.mvs_recorder is not None:
                self.mvs_recorder.write_frame(cam_id, frame, int(ts_us), int(self.frame_count[cam_id]), int(mvs_frame_num), mvs_meta)
            self.last_error[cam_id] = ""

            if time.time() - self._fps_t0 >= 1.0:
                dt = max(1e-6, time.time() - self._fps_t0)
                for cid in self.route_order:
                    self.fps_value[cid] = float(self._fps_count.get(cid, 0)) / dt
                    self._fps_count[cid] = 0
                self._fps_t0 = time.time()

    def get_latest(self, cam_id: str) -> Optional[Tuple[np.ndarray, int, int, int, Dict[str, Any], float]]:
        with self.lock:
            item = self.latest.get(cam_id)
            if item is None:
                return None
            frame, ts_us, fid, mvs_frame_num, mvs_meta, t = item
            return frame.copy(), int(ts_us), int(fid), int(mvs_frame_num), copy.deepcopy(mvs_meta), t

    def get_fps(self, cam_id: str) -> float:
        return float(self.fps_value.get(cam_id, 0.0))

    def get_error(self, cam_id: str) -> str:
        return str(self.last_error.get(cam_id, ""))

def _fit_for_display(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """等比例缩小显示完整画面，不裁剪。"""
    if img is None:
        return img
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    if max_w <= 0 and max_h <= 0:
        return img
    sw = float(max_w) / float(w) if max_w > 0 else 1.0
    sh = float(max_h) / float(h) if max_h > 0 else 1.0
    scale = min(sw, sh, 1.0)
    if scale >= 0.999:
        return img
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _draw_raw_mvs_preview(frame: np.ndarray, text: str) -> np.ndarray:
    """在 YOLO 推理前先显示 MVS 原始画面，避免 warmup 慢时看起来像 MVS 没画面。"""
    vis = frame.copy()
    try:
        cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 1, 760), 44), (0, 0, 0), -1)
        cv2.putText(vis, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    return vis


def _draw_waiting_mvs_preview(cam_id: str, args: argparse.Namespace, width: int, height: int) -> np.ndarray:
    w = max(320, min(int(width or 960), 960))
    h = max(180, min(int(height or 540), 540))
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:] = (24, 28, 32)
    serial = str(getattr(args, "serial", "") or "")
    lines = [
        f"MVS {cam_id} opened",
        f"serial={serial}",
        "waiting for first frame...",
    ]
    y = 54
    for line in lines:
        cv2.putText(vis, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        y += 42
    return vis


_IMSHOW_INITIALIZED_WINDOWS: set = set()


def _imshow_last_views(last_vis: Dict[str, np.ndarray], route_order: List[str], window_prefix: str, max_w: int, max_h: int) -> int:
    """显示每路最新画面。返回按键值；无窗口时返回 -1。"""
    key = -1
    for cam_id in route_order:
        if cam_id not in last_vis:
            continue
        win = f"{window_prefix}_{cam_id}"
        display_img = _fit_for_display(last_vis[cam_id], max_w, max_h)
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, display_img)
            if win not in _IMSHOW_INITIALIZED_WINDOWS:
                cv2.resizeWindow(win, display_img.shape[1], display_img.shape[0])
                _IMSHOW_INITIALIZED_WINDOWS.add(win)
        except Exception:
            cv2.imshow(win, display_img)
    try:
        key = cv2.waitKey(1) & 0xFF
    except Exception:
        key = -1
    return key


def _build_route_args_direct(default_args: argparse.Namespace, common_cfg: Dict[str, Any], route_cfg: Dict[str, Any], hik_defaults: Dict[str, Any]) -> argparse.Namespace:
    args = copy.deepcopy(default_args)
    _apply_sections_to_args(args, common_cfg, "direct.common")
    args.show = False
    args.source = str(route_cfg.get("source", args.source)).lower()
    args.resize_width = int(route_cfg.get("resize_width", 0))
    args.resize_height = int(route_cfg.get("resize_height", 0))

    if args.source == "hik":
        args.serial = str(route_cfg.get("serial", "")).strip()
        args.mvs_sdk_path = str(route_cfg.get("mvs_sdk_path", hik_defaults.get("mvs_sdk_path", getattr(args, "mvs_sdk_path", "/opt/MVS/Samples/64/Python/MvImport"))))
        args.hik_timeout_ms = int(route_cfg.get("hik_timeout_ms", hik_defaults.get("hik_timeout_ms", 1000)))
        # 0 表示不裁剪；配合 force_full_sensor=true 会主动恢复完整 ROI。
        args.capture_width = int(route_cfg.get("capture_width", hik_defaults.get("capture_width", 0)))
        args.capture_height = int(route_cfg.get("capture_height", hik_defaults.get("capture_height", 0)))
        args.capture_fps = float(route_cfg.get("capture_fps", hik_defaults.get("capture_fps", 25)))
        setattr(args, "force_full_sensor", bool(route_cfg.get("force_full_sensor", hik_defaults.get("force_full_sensor", True))))
        setattr(args, "prefer_bgr8", bool(route_cfg.get("prefer_bgr8", hik_defaults.get("prefer_bgr8", False))))
        setattr(args, "one_key_record_enable", _as_bool(route_cfg.get("one_key_record_enable", True), True))
    elif args.source == "video":
        args.video = str(route_cfg.get("video", route_cfg.get("path", "")))
        args.video_backend = str(route_cfg.get("backend", getattr(args, "video_backend", "auto"))).lower()
        setattr(args, "video_loop", bool(route_cfg.get("loop", True)))
        setattr(args, "video_realtime", bool(route_cfg.get("realtime", True)))
    else:
        raise ValueError(f"route source 只支持 hik/video，当前: {args.source}")

    calib_path = route_cfg.get("local_calib_path", route_cfg.get("calib_path", ""))
    if calib_path:
        args.local_calib_path = str(calib_path)
    return args


def parse_direct_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="精简多相机/多视频直显版本：独立窗口完整画面 + YOLO batch 推理")
    p.add_argument("--config", type=str, default=os.path.splitext(os.path.abspath(__file__))[0] + ".json")
    return p.parse_args()


def run_direct(cli_args: argparse.Namespace) -> None:
    cfg = load_direct_config(cli_args.config)
    print(f"[config] loaded: {cli_args.config}")
    print(f"[config] active_profile: {cfg.get('active_profile', '')}")

    common_cfg = cfg.get("common", {}) or {}
    hik_defaults = cfg.get("hik_defaults", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}
    display_cfg = cfg.get("display", {}) or {}
    ue5_headless_enable = _as_bool(runtime_cfg.get("ue5_headless_enable", False), False)
    parallel_postprocess_enable = _as_bool(runtime_cfg.get("parallel_postprocess_enable", ue5_headless_enable), ue5_headless_enable)
    event_overlay_cfg = cfg.get("event_overlay", display_cfg.get("event_overlay", {}) if isinstance(display_cfg, dict) else {}) or {}
    sync_cfg = cfg.get("sync", {}) or {}
    record_cfg = cfg.get("one_key_record", {}) or {}
    routes_cfg = [r for r in (cfg.get("routes", []) or []) if isinstance(r, dict) and bool(r.get("enabled", True))]
    if not routes_cfg:
        raise ValueError("配置中没有启用 routes")

    # 关键开关：当只需要验证 MVS+Event 融合是否流畅时，不加载 YOLO，也不做人检。
    # 这样显示刷新不再被 model.predict() 卡住，便于确认 MVS 采集是否真正丢帧。
    human_detection_enable = _as_bool(runtime_cfg.get("human_detection_enable", runtime_cfg.get("yolo_enable", True)), True)
    display_raw_fusion_every_mvs_frame = _as_bool(
        runtime_cfg.get("display_raw_fusion_every_mvs_frame", not human_detection_enable),
        not human_detection_enable,
    )
    write_empty_human_json_when_disabled = _as_bool(runtime_cfg.get("write_empty_human_json_when_disabled", False), False)

    # 250fps event-driven fusion mode:
    # - MVS remains 25fps/40ms and acts as the anchor background.
    # - Event overlay remains 250fps/4ms and drives preview refresh.
    # - Events within [MVS_k_anchor_ts, MVS_k_anchor_ts + 40ms) reuse MVS_k.
    event_driven_fusion_enable = _as_bool(runtime_cfg.get("event_driven_fusion_enable", False), False)
    event_driven_fusion_fps = float(runtime_cfg.get("event_driven_fusion_fps", runtime_cfg.get("fusion_display_fps", 250.0)) or 250.0)
    event_driven_fusion_fps = max(1.0, min(1000.0, event_driven_fusion_fps))
    event_driven_display_delay_ms = float(runtime_cfg.get("event_driven_display_delay_ms", 20.0) or 0.0)
    event_driven_anchor_period_ms = float(runtime_cfg.get("event_driven_anchor_period_ms", sync_cfg.get("mvs_soft_trigger_period_ms", 40.0)) or 40.0)
    event_driven_anchor_max_overshoot_ms = float(runtime_cfg.get("event_driven_anchor_max_overshoot_ms", 0.0) or 0.0)
    event_driven_require_strict_anchor = _as_bool(runtime_cfg.get("event_driven_require_strict_anchor", True), True)
    event_driven_anchor_history = int(runtime_cfg.get("event_driven_anchor_history", 64) or 64)
    event_driven_output_duplicate_ticks = _as_bool(runtime_cfg.get("event_driven_output_duplicate_ticks", True), True)
    if event_driven_fusion_enable and human_detection_enable:
        print("[event250][WARN] event_driven_fusion_enable=true is intended for human_detection_enable=false; disabling event-driven preview while YOLO is enabled", flush=True)
        event_driven_fusion_enable = False

    default_args = _make_base_default_args()
    common_args = copy.deepcopy(default_args)
    _apply_sections_to_args(common_args, common_cfg, "direct.common")
    device = str(getattr(common_args, "device", "") or "0")
    model = None
    if human_detection_enable:
        from ultralytics import YOLO
        model_arg, model_exists, model_abs = _resolve_yolo_model_path(common_args.model, cli_args.config)
        print(f"[model] loading {common_args.model}, device={device}, imgsz={common_args.imgsz}")
        print(f"[model] resolved_path={model_abs} exists={model_exists}")
        if model_exists:
            common_args.model = model_arg
        model = YOLO(common_args.model)
    else:
        print("[model] human_detection_enable=false: skip YOLO loading/predict; display MVS+Event fusion only", flush=True)

    route_order: List[str] = []
    route_args: Dict[str, argparse.Namespace] = {}
    soft_trigger_mode = _soft_trigger_enabled(sync_cfg)
    if soft_trigger_mode:
        print("[softsync] enabled: MVS TriggerMode=On + central 40ms TriggerSoftware loop", flush=True)
    workers: Dict[str, DirectFrameWorker] = {}
    soft_group: Optional[SoftTriggerCaptureGroup] = None
    managers: Dict[str, StrictStableIDManager] = {}
    projectors: Dict[str, LocalMapProjector] = {}
    a_anchor_reid_manager = AAnchorReIDManager(common_args)
    last_processed_fid: Dict[str, int] = {}
    # last_displayed_fid 专门控制“融合显示”刷新，和 last_processed_fid 的 YOLO 节流分开。
    # 关闭人检时：每个 MVS 新帧都会走显示；开启人检时：至少不会因为 process_every_n 跳过 raw/fusion 预览。
    last_displayed_fid: Dict[str, int] = {}
    last_vis: Dict[str, np.ndarray] = {}
    # Last true YOLO record per camera. Used to synthesize short-gap carry_last
    # human_result rows for missing MVS syncs.
    last_true_human_record: Dict[str, Dict[str, Any]] = {}
    # 记录每路是否已经有过 YOLO 处理后的画面。
    # 之前 raw preview 每轮推理前都会覆盖一次窗口，随后又被检测结果覆盖，
    # 会造成“标定点/叠加层一闪一闪”的视觉效果。
    has_processed_view: set = set()

    for r in routes_cfg:
        cam_id = str(r.get("id", r.get("name", f"cam{len(route_order) + 1}")))
        args = _build_route_args_direct(default_args, common_cfg, r, hik_defaults)
        args.device = device
        route_sync_cfg = copy.deepcopy(sync_cfg)
        if isinstance(r.get("sync"), dict):
            _deep_update(route_sync_cfg, r["sync"])
        for key in ("exposure_us", "gain", "exposure_auto_off", "gain_auto_off"):
            if key in r and r.get(key) is not None:
                route_sync_cfg[key] = copy.deepcopy(r.get(key))
        setattr(args, "sync_cfg", route_sync_cfg)
        print(
            f"[HIK][direct][sync][route={cam_id}] "
            f"final exposure_us={route_sync_cfg.get('exposure_us')} "
            f"gain={route_sync_cfg.get('gain')}",
            flush=True,
        )
        route_order.append(cam_id)
        route_args[cam_id] = args
        if not soft_trigger_mode:
            workers[cam_id] = DirectFrameWorker(cam_id, args)
        managers[cam_id] = StrictStableIDManager(args)
        projectors[cam_id] = LocalMapProjector(args)
        last_processed_fid[cam_id] = 0
        last_displayed_fid[cam_id] = 0
        print(f"[route][{cam_id}] source={args.source}, serial={getattr(args,'serial','')}, video={getattr(args,'video','')}, capture={getattr(args,'capture_width',0)}x{getattr(args,'capture_height',0)}@{getattr(args,'capture_fps',0)}, full_roi={getattr(args,'force_full_sensor', True)}, prefer_bgr8={getattr(args,'prefer_bgr8', False)}, native_sampling=1x, resize={getattr(args,'resize_width',0)}x{getattr(args,'resize_height',0)}")

    _wait_for_event_ready_files(sync_cfg, route_order, route_args)
    if human_detection_enable or write_empty_human_json_when_disabled:
        human_writers = _open_human_jsonl_writers(sync_cfg, route_order, route_args)
    else:
        human_writers = {}
        print("[human_jsonl] disabled because human_detection_enable=false", flush=True)

    # YOLO batch performance CSV writer.
    # It must be initialized before the main loop because the loop writes one row
    # after each model.predict() batch.  Keep it as None when human detection is
    # disabled so the rest of the code can safely check it.
    yolo_perf_writer = YoloInferPerfWriter(sync_cfg) if human_detection_enable else None

    mvs_recorder = MvsRawRecorderManager(record_cfg, sync_cfg, route_order, route_args)
    mvs_recorder.start()

    if soft_trigger_mode:
        soft_group = SoftTriggerCaptureGroup(route_order, route_args, sync_cfg, mvs_recorder=mvs_recorder)
        soft_group.start()
    else:
        for w in workers.values():
            w.start()

    max_batch = int(runtime_cfg.get("max_batch", len(route_order)) or len(route_order))
    max_batch = max(1, min(max_batch, max(1, len(route_order))))
    # TensorRT engine is fast enough that under-filled batches become the main
    # coverage limiter. After the first scan, wait a few ms for slower routes so
    # the YOLO batch is closer to one frame per camera. This improves per-sync
    # human_result coverage without changing model accuracy.
    yolo_batch_collect_wait_ms = float(runtime_cfg.get("yolo_batch_collect_wait_ms", 5.0) or 0.0)
    yolo_batch_collect_sleep_ms = float(runtime_cfg.get("yolo_batch_collect_sleep_ms", 0.5) or 0.5)
    idle_sleep = float(runtime_cfg.get("idle_sleep_sec", 0.002))
    print_interval = float(runtime_cfg.get("print_interval_sec", 1.0))
    print(
        f"[runtime] human_detection_enable={human_detection_enable} "
        f"display_raw_fusion_every_mvs_frame={display_raw_fusion_every_mvs_frame} "
        f"event_driven_fusion_enable={event_driven_fusion_enable} "
        f"event_driven_fusion_fps={event_driven_fusion_fps:.1f} "
        f"anchor_period_ms={event_driven_anchor_period_ms:.1f} "
        f"display_delay_ms={event_driven_display_delay_ms:.1f} "
        f"max_batch={max_batch} routes={len(route_order)} "
        f"yolo_batch_collect_wait_ms={yolo_batch_collect_wait_ms:.2f} "
        f"collect_sleep_ms={yolo_batch_collect_sleep_ms:.2f}",
        flush=True,
    )
    # 关键补丁：先显示 MVS 原始画面，再进入 YOLO 推理。
    # 这样 YOLO 第一次 warmup 很慢时，用户也能立即看到 MVS 画面，确认相机不是没打开。
    show_raw_before_infer = _as_bool(runtime_cfg.get("show_raw_before_infer", True), True)
    # true：只在某路相机还没有第一张 YOLO 结果前显示 raw preview；
    # false：保持旧行为，每次推理前都显示 raw preview。
    # 建议保持 true，避免 raw 画面和叠加画面交替导致闪烁。
    raw_preview_until_first_result = _as_bool(runtime_cfg.get("raw_preview_until_first_result", True), True)
    print_first_raw_preview = _as_bool(runtime_cfg.get("print_first_raw_preview", True), True)
    raw_preview_printed: set = set()
    display_max_w = int(display_cfg.get("max_width", 960))
    display_max_h = int(display_cfg.get("max_height", 720))
    window_prefix = str(display_cfg.get("window_prefix", "RGB_YOLO_SEG_DIRECT"))
    show = bool(display_cfg.get("show", True))
    if ue5_headless_enable:
        show = False
        show_raw_before_infer = False
        raw_preview_until_first_result = True
        print("[ue5_headless] local OpenCV preview disabled; MVS shm + human_result JSONL remain enabled", flush=True)
    if isinstance(event_overlay_cfg, dict) and not event_overlay_cfg.get("_save_path"):
        event_overlay_cfg["_save_path"] = os.path.abspath(cli_args.config)
    # When an external Qt/OpenGL renderer is used, the MVS process only needs to
    # publish raw MVS frames.  Avoid the old CPU event-overlay reader here so the
    # event shared memory is not consumed twice and OpenCV fusion is bypassed.
    event_overlay_manager_cfg = copy.deepcopy(event_overlay_cfg) if isinstance(event_overlay_cfg, dict) else {}
    if _as_bool(runtime_cfg.get("use_opengl_renderer", False), False) and _as_bool(runtime_cfg.get("opengl_skip_cpu_event_overlay", True), True):
        event_overlay_manager_cfg["enabled"] = False
        print("[opengl_renderer] CPU EventSharedOverlayManager disabled in MVS process; renderer reads event shm directly", flush=True)
    event_overlay = EventSharedOverlayManager(event_overlay_manager_cfg, sync_cfg=sync_cfg)
    if show:
        for cam_id in route_order:
            last_vis[cam_id] = _draw_waiting_mvs_preview(cam_id, route_args[cam_id], display_max_w, display_max_h)
        _imshow_last_views(last_vis, route_order, window_prefix, display_max_w, display_max_h)
        print(f"[preview] initialized MVS preview windows for routes: {route_order}", flush=True)
    event_overlay.start_history(route_order)

    postprocess_pool = None
    if parallel_postprocess_enable:
        postprocess_workers = max(1, min(len(route_order), int(runtime_cfg.get("parallel_postprocess_workers", len(route_order)) or len(route_order))))
        postprocess_pool = concurrent.futures.ThreadPoolExecutor(max_workers=postprocess_workers, thread_name_prefix="human_post")
        print(f"[ue5_headless] parallel postprocess enabled workers={postprocess_workers}", flush=True)

    # Per-route MVS anchor buffers for 250fps event-driven fusion.
    # Each anchor stores one 25fps MVS frame plus the event-clock timestamp of the
    # corresponding MVS trigger. Event overlay frames are assigned to the newest
    # anchor whose anchor_event_ts_us <= event_ts_us < anchor_event_ts_us + 40ms.
    mvs_anchor_buffers: Dict[str, deque] = {cid: deque(maxlen=max(2, event_driven_anchor_history)) for cid in route_order}
    last_anchor_fid: Dict[str, int] = {cid: 0 for cid in route_order}
    last_event_driven_event_fid: Dict[str, int] = {cid: -1 for cid in route_order}
    next_event_driven_tick = time.perf_counter()

    def _update_event_driven_anchor(cam_id: str, item: Tuple[np.ndarray, int, int, int, Dict[str, Any], Any]) -> None:
        frame, ts_us, fid, mvs_frame_num, mvs_meta, _ = item
        if int(fid) == int(last_anchor_fid.get(cam_id, 0)):
            return
        anchor_info = event_overlay.resolve_mvs_anchor_time(
            cam_id,
            mvs_ts_us=int(ts_us or 0),
            mvs_frame_num=int(mvs_frame_num or 0),
            mvs_meta=mvs_meta if isinstance(mvs_meta, dict) else {},
        )
        # Strict event-driven fusion must anchor each MVS frame from the shared
        # 40ms trigger log only. If the event_trigger CSV line has not arrived yet,
        # resolve_mvs_anchor_time() may return a non-strict fallback timestamp; do
        # not create an anchor from it in this mode. Also keep last_anchor_fid
        # unchanged, so the same latest MVS frame is retried on later loop ticks
        # after the matching CSV row becomes available.
        if event_driven_fusion_enable and event_driven_require_strict_anchor and not bool(anchor_info.get("strict", False)):
            return
        anchor_ts = int(anchor_info.get("anchor_event_ts_us", 0) or 0)
        if anchor_ts <= 0:
            return
        anchor = {
            "frame": np.asarray(frame).copy(),
            "ts_us": int(ts_us or 0),
            "fid": int(fid),
            "mvs_frame_num": int(mvs_frame_num or 0),
            "mvs_meta": copy.deepcopy(mvs_meta or {}),
            "anchor_event_ts_us": anchor_ts,
            "mvs_sync_index": anchor_info.get("mvs_sync_index"),
            "anchor_source": str(anchor_info.get("source", "")),
            "strict": bool(anchor_info.get("strict", False)),
        }
        buf = mvs_anchor_buffers.setdefault(cam_id, deque(maxlen=max(2, event_driven_anchor_history)))
        # Avoid duplicate anchors if the same timestamp is reported twice.
        if not buf or int(buf[-1].get("fid", -1)) != int(fid):
            buf.append(anchor)
            last_anchor_fid[cam_id] = int(fid)

    def _select_event_driven_anchor(cam_id: str, event_ts_us: int) -> Optional[Dict[str, Any]]:
        buf = list(mvs_anchor_buffers.get(cam_id, deque()))
        if not buf:
            return None
        if int(event_ts_us or 0) <= 0:
            return buf[-1]
        period_us = int(max(1.0, event_driven_anchor_period_ms) * 1000.0)
        overshoot_us = int(max(0.0, event_driven_anchor_max_overshoot_ms) * 1000.0)
        selected = None
        for anchor in buf:
            a_ts = int(anchor.get("anchor_event_ts_us", 0) or 0)
            if a_ts <= int(event_ts_us):
                selected = anchor
            else:
                break
        if selected is None:
            return None
        start_ts = int(selected.get("anchor_event_ts_us", 0) or 0)
        if int(event_ts_us) >= start_ts + period_us + overshoot_us:
            # This event already belongs to the next 40ms MVS period, but the next MVS
            # frame has not been delivered/anchored yet. Skip instead of pairing it with
            # the wrong old MVS background. Increase event_driven_display_delay_ms if this
            # happens often around boundaries.
            return None
        return selected

    def _handle_preview_key(key: int) -> bool:
        if key < 0:
            return False
        if event_overlay.handle_key(key, route_order):
            return False
        if key in (27, ord("q")):
            return True
        if key in (ord("s"), ord("S")):
            if _record_cfg_enabled(record_cfg):
                cmd_file = _record_command_file(record_cfg, sync_cfg)
                if mvs_recorder.is_recording():
                    write_one_key_record_command(cmd_file, "stop")
                else:
                    write_one_key_record_command(cmd_file, "start", session_id=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            else:
                print("[onekey_record] disabled in human JSON one_key_record.enable=false", flush=True)
            return False
        if key in (ord("p"), ord("P")):
            snap_dir = str(display_cfg.get("snapshot_dir", "") or os.getcwd())
            os.makedirs(snap_dir, exist_ok=True)
            for snap_cam_id, img in last_vis.items():
                path = os.path.join(snap_dir, f"{snap_cam_id}_{int(time.time())}.png")
                cv2.imwrite(path, img)
                print(f"[snapshot][{snap_cam_id}] {path}")
            return False
        return False

    def _event_driven_fusion_tick() -> Tuple[bool, bool]:
        """Render one 250fps event-driven fusion tick. Returns (updated, should_break)."""
        nonlocal display_frame_count
        updated = False
        for cam_id in route_order:
            event_item = event_overlay.capture_event_driven_item(cam_id, display_delay_ms=event_driven_display_delay_ms)
            if event_item is None:
                continue
            event_fid = int(event_item.get("frame_id", -1) or -1)
            if (not event_driven_output_duplicate_ticks) and event_fid == int(last_event_driven_event_fid.get(cam_id, -2)):
                continue
            event_ts = int(event_item.get("timestamp_us", 0) or 0)
            anchor = _select_event_driven_anchor(cam_id, event_ts)
            if anchor is None:
                continue
            offset_ms = (float(event_ts) - float(anchor.get("anchor_event_ts_us", 0) or 0)) / 1000.0
            frame = np.asarray(anchor["frame"]).copy()
            vis = event_overlay.apply_item(cam_id, frame, event_item)
            sync_label = anchor.get("mvs_sync_index")
            strict_label = "STRICT" if bool(anchor.get("strict", False)) else "FALLBACK"
            raw_text = (
                f"{cam_id} EVENT250 {strict_label} "
                f"MVS={anchor.get('mvs_frame_num')} SYNC={sync_label} "
                f"EVT={event_fid} offset={offset_ms:.1f}ms "
                f"src={anchor.get('anchor_source')}"
            )
            last_vis[cam_id] = _draw_raw_mvs_preview(vis, raw_text)
            last_event_driven_event_fid[cam_id] = event_fid
            display_frame_count += 1
            updated = True
        should_break = False
        if show and updated:
            key = _imshow_last_views(last_vis, route_order, window_prefix, display_max_w, display_max_h)
            should_break = _handle_preview_key(key)
        return updated, should_break

    def _try_add_yolo_batch_item(
        cam_id: str,
        frames: List[np.ndarray],
        metas: List[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace, Optional[Dict[str, Any]]]],
        collected_cam_ids: set,
    ) -> bool:
        """Try to add the latest unprocessed frame for one camera to the current YOLO batch.

        The main loop previously scanned each camera once and immediately launched
        model.predict() if any frame was ready. With a fast TensorRT engine this often
        produced 3-frame batches for a 5-camera system, lowering per-sync coverage.
        This helper lets the loop rescan briefly and fill the missing cameras.
        """
        if cam_id in collected_cam_ids:
            return False
        if soft_trigger_mode:
            item = soft_group.get_latest(cam_id) if soft_group is not None else None
        else:
            item = workers[cam_id].get_latest()
        if item is None:
            return False
        frame, ts_us, fid, mvs_frame_num, mvs_meta, _ = item
        args = route_args[cam_id]
        last_fid = last_processed_fid.get(cam_id, 0)
        if fid == last_fid:
            return False
        if fid - last_fid < int(args.process_every_n):
            return False
        if event_driven_fusion_enable:
            _update_event_driven_anchor(cam_id, item)
        event_item = event_overlay.capture_for_mvs_frame(
            cam_id,
            mvs_ts_us=ts_us,
            mvs_frame_num=mvs_frame_num,
            mvs_meta=mvs_meta,
        )
        frames.append(frame)
        metas.append((cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args, event_item))
        collected_cam_ids.add(cam_id)
        return True

    last_print = time.time()
    infer_batch_count = 0
    infer_frame_count = 0
    display_frame_count = 0
    last_infer_ms = 0.0
    yolo_batch_index = 0
    route_scan_offset = 0

    try:
        while True:
            frames: List[np.ndarray] = []
            metas: List[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace, Optional[Dict[str, Any]]]] = []
            collected_cam_ids: set = set()
            batch_collect_wait_actual_ms = 0.0
            display_updated = False
            scan_order = route_order[route_scan_offset:] + route_order[:route_scan_offset]
            route_scan_offset = (route_scan_offset + 1) % max(1, len(route_order))
            for cam_id in scan_order:
                if soft_trigger_mode:
                    item = soft_group.get_latest(cam_id) if soft_group is not None else None
                else:
                    item = workers[cam_id].get_latest()
                if item is None:
                    continue
                frame, ts_us, fid, mvs_frame_num, mvs_meta, _ = item
                args = route_args[cam_id]

                if event_driven_fusion_enable:
                    _update_event_driven_anchor(cam_id, item)

                # 显示路径与 YOLO 节流分开：只要是新的 MVS 帧，就可以先显示 MVS+Event 融合。
                # 关闭人检时，这条路径就是主路径；开启人检时，它也能避免 process_every_n 本身造成“显示层跳帧”。
                event_item_for_frame: Optional[Dict[str, Any]] = None
                if (display_raw_fusion_every_mvs_frame and (not event_driven_fusion_enable)
                        and fid != last_displayed_fid.get(cam_id, 0)):
                    event_item_for_frame = event_overlay.capture_for_mvs_frame(
                        cam_id,
                        mvs_ts_us=ts_us,
                        mvs_frame_num=mvs_frame_num,
                        mvs_meta=mvs_meta,
                    )
                    if show:
                        cap_fps = soft_group.get_fps(cam_id) if soft_trigger_mode and soft_group is not None else workers[cam_id].fps_value
                        sync_label = mvs_meta.get("program_sync_index", "-") if isinstance(mvs_meta, dict) else "-"
                        mode_text = "HUMAN=OFF" if not human_detection_enable else "YOLO=async/display-decoupled"
                        raw_text = f"{cam_id} FUSED MVS+EVENT {mode_text} cap={cap_fps:.1f} MVS={mvs_frame_num} SYNC={sync_label}"
                        raw_vis = event_overlay.apply_item(cam_id, frame, event_item_for_frame)
                        last_vis[cam_id] = _draw_raw_mvs_preview(raw_vis, raw_text)
                    last_displayed_fid[cam_id] = fid
                    display_frame_count += 1
                    display_updated = True

                    # 可选：人检关闭但仍输出空 human_result，方便下游保持“无人体”状态。默认不写，避免额外 IO。
                    if (not human_detection_enable) and human_writers:
                        rec = _build_human_sync_record(
                            cam_id=cam_id,
                            args=args,
                            sync_cfg=sync_cfg,
                            fid=fid,
                            mvs_frame_num=mvs_frame_num,
                            mvs_meta=mvs_meta,
                            ts_us=ts_us,
                            frame_shape=frame.shape,
                            persons=[],
                        )
                        _append_human_sync_record(
                            human_writers,
                            cam_id,
                            rec,
                            flush=_as_bool(sync_cfg.get("flush_every_frame", True), True),
                        )

                if not human_detection_enable:
                    # 关闭人检时，不再受 process_every_n / max_batch / model.predict 限制。
                    last_processed_fid[cam_id] = fid
                    continue

                last_fid = last_processed_fid.get(cam_id, 0)
                if fid == last_fid:
                    continue
                if fid - last_fid < int(args.process_every_n):
                    continue
                # 关键时序修正：MVS 帧刚被选入本轮 YOLO batch 时，立刻捕获/匹配事件帧，
                # 并把 event_item 和这张 MVS 帧绑定。后面 YOLO 推理无论耗时多久，
                # 显示融合都不再重新读取 event_overlay_latest 的“未来最新帧”。
                event_item = None if ue5_headless_enable else event_item_for_frame
                if (not ue5_headless_enable) and event_item is None:
                    event_item = event_overlay.capture_for_mvs_frame(
                        cam_id,
                        mvs_ts_us=ts_us,
                        mvs_frame_num=mvs_frame_num,
                        mvs_meta=mvs_meta,
                    )
                frames.append(frame)
                metas.append((cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args, event_item))
                collected_cam_ids.add(cam_id)
                if len(frames) >= max_batch:
                    break

            # Engine optimization: when a partial batch is available, wait a very
            # short time for slower camera receiver threads so the batch is closer
            # to 5 cameras. This targets avg_batch, not model speed.
            if frames and human_detection_enable and len(frames) < max_batch and yolo_batch_collect_wait_ms > 0.0:
                _collect_t0 = time.perf_counter()
                _deadline = _collect_t0 + max(0.0, yolo_batch_collect_wait_ms) / 1000.0
                while len(frames) < max_batch:
                    _now = time.perf_counter()
                    if _now >= _deadline:
                        break
                    _added = False
                    for _cid in route_order:
                        if len(frames) >= max_batch:
                            break
                        if _try_add_yolo_batch_item(_cid, frames, metas, collected_cam_ids):
                            _added = True
                    if len(frames) >= max_batch:
                        break
                    if not _added:
                        _remain = max(0.0, _deadline - time.perf_counter())
                        time.sleep(min(max(0.0001, yolo_batch_collect_sleep_ms / 1000.0), _remain))
                batch_collect_wait_actual_ms = (time.perf_counter() - _collect_t0) * 1000.0

            if event_driven_fusion_enable:
                tick_dt = 1.0 / max(1.0, float(event_driven_fusion_fps))
                now_perf = time.perf_counter()
                if now_perf >= next_event_driven_tick:
                    _updated_250, _break_250 = _event_driven_fusion_tick()
                    if _break_250:
                        break
                    next_event_driven_tick += tick_dt
                    if next_event_driven_tick < now_perf - tick_dt * 2.0:
                        # Do not try to catch up by rendering a burst of old frames after a stall.
                        next_event_driven_tick = now_perf + tick_dt
                else:
                    time.sleep(max(0.0005, min(idle_sleep, next_event_driven_tick - now_perf)))

            if frames and human_detection_enable:
                # 先把最新 MVS 原始图显示出来。之前的版本必须等 YOLO 第一轮 predict 完成后才 imshow，
                # 所以 warmup 卡住时看起来像“MVS 没画面”。这里先 raw preview，再做人检。
                if show and show_raw_before_infer and (not display_raw_fusion_every_mvs_frame):
                    for meta in metas:
                        cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args, event_item = meta
                        # 如果该相机已经有过检测后的叠加画面，就不再用 raw preview 覆盖窗口。
                        # 否则窗口会在“无标定点 raw 图”和“有标定点/检测叠加图”之间快速切换，看起来像白点闪烁/跳动。
                        if raw_preview_until_first_result and cam_id in has_processed_view:
                            continue
                        raw_text = f"{cam_id} RAW MVS frame={mvs_frame_num} local_fid={fid}  YOLO warming/running..."
                        raw_vis = event_overlay.apply_item(cam_id, frame, event_item)
                        last_vis[cam_id] = _draw_raw_mvs_preview(raw_vis, raw_text)
                        if print_first_raw_preview and cam_id not in raw_preview_printed:
                            raw_preview_printed.add(cam_id)
                            print(f"[preview][{cam_id}] first raw MVS preview shown: shape={frame.shape}, mvs_frame_num={mvs_frame_num}", flush=True)
                    key = _imshow_last_views(last_vis, route_order, window_prefix, display_max_w, display_max_h)
                    if event_overlay.handle_key(key, route_order):
                        pass
                    elif key in (27, ord("q")):
                        break

                batch_wall_start = time.time()
                kwargs = _build_yolo_predict_kwargs(frames, common_args, device=device)
                if model is None:
                    raise RuntimeError("human_detection_enable=true but YOLO model is not loaded")
                t_predict = time.time()
                results = model.predict(**kwargs)
                predict_ms = (time.time() - t_predict) * 1000.0
                last_infer_ms = predict_ms
                infer_batch_count += 1
                infer_frame_count += len(frames)
                yolo_batch_index += 1
                batch_person_counts: List[int] = []
                pending_human_records: List[Tuple[str, Dict[str, Any]]] = []
                batch_post_start = time.time()

                processed_items: List[Tuple[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace], List[PersonInstance]]] = []
                processed_views: List[Tuple[Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace, Any], List[PersonInstance]]] = []

                def _postprocess_one(_idx: int, _result: Any, _meta: Tuple[str, int, int, int, Dict[str, Any], np.ndarray, argparse.Namespace, Any]):
                    _cam_id, _ts_us, _fid, _mvs_frame_num, _mvs_meta, _frame, _args, _event_item = _meta
                    _persons = extract_person_instances(_result, _frame.shape, _args)
                    assign_local_positions(_persons, projectors[_cam_id], _args)
                    _persons = managers[_cam_id].update(_persons, _frame, _ts_us, _fid)
                    return _idx, _meta, _persons

                if parallel_postprocess_enable and postprocess_pool is not None and len(metas) > 1:
                    futures = [postprocess_pool.submit(_postprocess_one, i, result, meta) for i, (result, meta) in enumerate(zip(results, metas))]
                    post_results = [f.result() for f in futures]
                    post_results.sort(key=lambda x: int(x[0]))
                else:
                    post_results = [_postprocess_one(i, result, meta) for i, (result, meta) in enumerate(zip(results, metas))]

                for _idx, meta, persons in post_results:
                    cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args, event_item = meta
                    batch_person_counts.append(int(len(persons)))
                    processed_items.append(((cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args), persons))
                    processed_views.append((meta, persons))

                # A-anchor OSNet ReID: A 相机建 P1/P2/P3 身份库，B/C/D/E 只匹配 A。
                # 这里只写 global_id_from_A / reid_score，不复用旧 mask/bbox；mask 仍然来自当前帧 YOLO-seg。
                a_anchor_reid_manager.assign_batch(processed_items)
                fusion_diag = apply_topdown_global_person_fusion(processed_items)

                for meta, persons in processed_views:
                    cam_id, ts_us, fid, mvs_frame_num, mvs_meta, frame, args, event_item = meta
                    if human_writers:
                        # Fill final batch timings after the loop; these values are still useful per-frame
                        # because all frames in the batch shared one model.predict() call.
                        infer_meta = {
                            "infer_source": "yolo",
                            "batch_index": int(yolo_batch_index),
                            "batch_size": int(len(frames)),
                            "collect_wait_ms": round(float(batch_collect_wait_actual_ms), 3),
                            "predict_ms": round(float(predict_ms), 3),
                            "topdown_fusion": fusion_diag,
                        }
                        rec = _build_human_sync_record(
                            cam_id=cam_id,
                            args=args,
                            sync_cfg=sync_cfg,
                            fid=fid,
                            mvs_frame_num=mvs_frame_num,
                            mvs_meta=mvs_meta,
                            ts_us=ts_us,
                            frame_shape=frame.shape,
                            persons=persons,
                            infer_meta=infer_meta,
                        )
                        pending_human_records.append((cam_id, rec))
                    if not ue5_headless_enable:
                        cap_fps = soft_group.get_fps(cam_id) if soft_trigger_mode and soft_group is not None else workers[cam_id].fps_value
                        sync_label = mvs_meta.get("program_sync_index", "-") if isinstance(mvs_meta, dict) else "-"
                        fps_text = f"{cam_id} cap={cap_fps:.1f} infer={last_infer_ms:.1f}ms P={len(persons)} shape={frame.shape[1]}x{frame.shape[0]} MVS={mvs_frame_num} SYNC={sync_label}"
                        vis = draw_instances(frame, persons, args, fps_text=fps_text)
                        draw_local_region(vis, projectors[cam_id], args)
                        vis = event_overlay.apply_item(cam_id, vis, event_item)
                        last_vis[cam_id] = vis
                        has_processed_view.add(cam_id)
                        last_displayed_fid[cam_id] = fid
                    last_processed_fid[cam_id] = fid
                postprocess_batch_ms = (time.time() - batch_post_start) * 1000.0
                total_batch_ms = (time.time() - batch_wall_start) * 1000.0
                for _cam_id, _rec in pending_human_records:
                    _rec["yolo_postprocess_batch_ms"] = round(float(postprocess_batch_ms), 3)
                    _rec["yolo_total_batch_ms"] = round(float(total_batch_ms), 3)
                    _flush_human = _as_bool(sync_cfg.get("flush_every_frame", True), True)
                    # Fill small missing sync gaps with explicitly marked carry_last rows
                    # before writing the current true YOLO row. This makes display/hit
                    # judgement continuous without pretending the carried row is a new
                    # detector output.
                    _emit_human_carry_gap_records(
                        human_writers,
                        _cam_id,
                        last_true_human_record.get(_cam_id),
                        _rec,
                        sync_cfg,
                        flush=_flush_human,
                    )
                    _append_human_sync_record(
                        human_writers,
                        _cam_id,
                        _rec,
                        flush=_flush_human,
                    )
                    last_true_human_record[_cam_id] = copy.deepcopy(_rec)
                if yolo_perf_writer is not None:
                    yolo_perf_writer.write(
                        batch_index=yolo_batch_index,
                        common_args=common_args,
                        device=device,
                        metas=metas,
                        person_counts=batch_person_counts,
                        predict_ms=predict_ms,
                        postprocess_batch_ms=postprocess_batch_ms,
                        total_batch_ms=total_batch_ms,
                        collect_wait_ms=batch_collect_wait_actual_ms,
                    )
            elif not display_updated:
                time.sleep(idle_sleep)


            if (not soft_trigger_mode) and all((route_args[cid].source == "video" and workers[cid].ended) for cid in route_order):
                print("[done] all video routes ended")
                break

            if show and (not event_driven_fusion_enable):
                key = _imshow_last_views(last_vis, route_order, window_prefix, display_max_w, display_max_h)
                if _handle_preview_key(key):
                    break

            if time.time() - last_print >= print_interval:
                dt = max(1e-6, time.time() - last_print)
                if soft_trigger_mode and soft_group is not None:
                    cap_info = " ".join([f"{cid}:{soft_group.get_fps(cid):.1f}" for cid in route_order])
                    active_n = sum(1 for cid in route_order if not soft_group.get_error(cid))
                    print(f"[perf][softsync] active={active_n}/{len(route_order)} display_frames/s={display_frame_count/dt:.1f} batch/s={infer_batch_count/dt:.1f} infer_frames/s={infer_frame_count/dt:.1f} last_infer_ms={last_infer_ms:.1f} cap_fps[{cap_info}]")
                    for cid in route_order:
                        err = soft_group.get_error(cid)
                        if err:
                            print(f"[capture_error][{cid}] {err}")
                else:
                    cap_info = " ".join([f"{cid}:{workers[cid].fps_value:.1f}" for cid in route_order])
                    print(f"[perf] active={sum(1 for w in workers.values() if not w.last_error)}/{len(route_order)} display_frames/s={display_frame_count/dt:.1f} batch/s={infer_batch_count/dt:.1f} infer_frames/s={infer_frame_count/dt:.1f} last_infer_ms={last_infer_ms:.1f} cap_fps[{cap_info}]")
                    for cid, w in workers.items():
                        if w.last_error:
                            print(f"[capture_error][{cid}] {w.last_error}")
                infer_batch_count = 0
                infer_frame_count = 0
                display_frame_count = 0
                last_print = time.time()
    finally:
        if soft_trigger_mode and soft_group is not None:
            soft_group.stop()
        else:
            for w in workers.values():
                w.stop()
        if 'mvs_recorder' in locals() and mvs_recorder is not None:
            mvs_recorder.close()
        if 'postprocess_pool' in locals() and postprocess_pool is not None:
            try:
                postprocess_pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                postprocess_pool.shutdown(wait=True)
        if 'event_overlay' in locals() and event_overlay is not None:
            event_overlay.close()
        _close_jsonl_writers(human_writers if 'human_writers' in locals() else {})
        if 'yolo_perf_writer' in locals() and yolo_perf_writer is not None:
            yolo_perf_writer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_direct(parse_direct_args())
