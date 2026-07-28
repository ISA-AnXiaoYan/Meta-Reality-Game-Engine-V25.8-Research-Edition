#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_homography(cfg: dict, cam: str):
    per_route = cfg["mvs_config"]["event_overlay"]["per_route"]
    if cam not in per_route:
        raise KeyError(f"配置里找不到 mvs_config.event_overlay.per_route.{cam}")

    route = per_route[cam]
    active = route.get("active_homography", "flight")

    homos = route.get("homographies_event_to_mvs", {})
    if active not in homos:
        raise KeyError(
            f"配置里找不到 {cam} 的 homography: "
            f"homographies_event_to_mvs.{active}"
        )

    return active, homos[active]


def apply_homography(H, x, y):
    h00, h01, h02 = H[0]
    h10, h11, h12 = H[1]
    h20, h21, h22 = H[2]

    z = h20 * x + h21 * y + h22
    if abs(z) < 1e-9:
        return None

    mx = (h00 * x + h01 * y + h02) / z
    my = (h10 * x + h11 * y + h12) / z
    return mx, my


def read_mvs_size(sync_root: Path, cam: str, default_w: int, default_h: int):
    meta_path = sync_root / f"mvs_latest_{cam}.json"
    if not meta_path.exists():
        return default_w, default_h, f"未找到 {meta_path}，使用默认 MVS 尺寸"

    try:
        meta = load_json(meta_path)
        shape = meta.get("shape")

        # shape 通常是 [height, width, channels]
        if isinstance(shape, list) and len(shape) >= 2:
            h = int(shape[0])
            w = int(shape[1])
            if w > 0 and h > 0:
                return w, h, f"从 {meta_path} 读取 MVS 尺寸"

        w = meta.get("width")
        h = meta.get("height")
        if w and h:
            return int(w), int(h), f"从 {meta_path} 读取 MVS 尺寸"

    except Exception as e:
        return default_w, default_h, f"读取 {meta_path} 失败：{e}，使用默认 MVS 尺寸"

    return default_w, default_h, f"{meta_path} 中没有有效尺寸，使用默认 MVS 尺寸"


def write_segment(
    f,
    cam: str,
    sync_run_id: str,
    bullet_id: int,
    segment_index: int,
    p0,
    p1,
    mvs_w: int,
    mvs_h: int,
    source: str,
):
    ts = int(time.time() * 1_000_000)

    x0, y0 = p0
    x1, y1 = p1

    msg0 = {
        "type": "overlay_bullet_point",
        "schema_version": 2,
        "sync_run_id": sync_run_id,
        "pair_id": cam,
        "camera_alias": cam,
        "camera_sn": "debug_event_fov",
        "bullet_id": bullet_id,
        "bullet_seq_id": 0,
        "point_index": 0,
        "point_ts_us": ts,
        "x": x0,
        "y": y0,
        "prev_x": x0,
        "prev_y": y0,
        "prev_valid": False,
        "display_width": mvs_w,
        "display_height": mvs_h,
        "confidence": 1.0,
        "status": "tracking",
        "state": "track",
        "semantic_state": "track",
        "source": source,
    }

    msg1 = {
        "type": "overlay_bullet_point",
        "schema_version": 2,
        "sync_run_id": sync_run_id,
        "pair_id": cam,
        "camera_alias": cam,
        "camera_sn": "debug_event_fov",
        "bullet_id": bullet_id,
        "bullet_seq_id": 1,
        "point_index": 1,
        "point_ts_us": ts + 4000,
        "x": x1,
        "y": y1,
        "prev_x": x0,
        "prev_y": y0,
        "prev_valid": True,
        "prev_point_index": 0,
        "prev_point_ts_us": ts,
        "display_width": mvs_w,
        "display_height": mvs_h,
        "confidence": 1.0,
        "status": "tracking",
        "state": "track",
        "semantic_state": "track",
        "point_delta_ms": 4.0,
        "source": source,
        "debug_segment_index": segment_index,
    }

    f.write(json.dumps(msg0, ensure_ascii=False, separators=(",", ":")) + "\n")
    f.write(json.dumps(msg1, ensure_ascii=False, separators=(",", ":")) + "\n")
    f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Draw event-camera FOV projected into MVS coordinates and send to UE5 via overlay_bullet_point jsonl."
    )
    parser.add_argument("--config", default="ids_8cam_fusion_config.json")
    parser.add_argument("--sync-root", default="sync_ipc")
    parser.add_argument("--cam", default="A")
    parser.add_argument("--event-w", type=int, default=1280)
    parser.add_argument("--event-h", type=int, default=720)
    parser.add_argument("--mvs-w", type=int, default=1624)
    parser.add_argument("--mvs-h", type=int, default=1240)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--loop-sleep", type=float, default=0.15)
    parser.add_argument("--draw-diagonal", action="store_true")
    args = parser.parse_args()

    cam = args.cam.strip().upper()
    cfg_path = Path(args.config)
    sync_root = Path(args.sync_root)

    cfg = load_json(cfg_path)
    active_name, H = get_homography(cfg, cam)

    mvs_w, mvs_h, size_note = read_mvs_size(
        sync_root,
        cam,
        args.mvs_w,
        args.mvs_h,
    )

    event_w = args.event_w
    event_h = args.event_h

    event_corners = [
        ("LT", 0.0, 0.0),
        ("RT", float(event_w), 0.0),
        ("RB", float(event_w), float(event_h)),
        ("LB", 0.0, float(event_h)),
    ]

    mvs_corners = []
    for name, ex, ey in event_corners:
        out = apply_homography(H, ex, ey)
        if out is None:
            raise RuntimeError(f"Homography 投影失败：{name} ({ex}, {ey}) z 接近 0")
        mx, my = out
        mvs_corners.append((name, mx, my))

    print("=" * 72)
    print(f"Config       : {cfg_path}")
    print(f"Camera       : {cam}")
    print(f"Homography   : {active_name}")
    print(f"Event size   : {event_w} x {event_h}")
    print(f"MVS size     : {mvs_w} x {mvs_h}    ({size_note})")
    print("-" * 72)
    print("事件相机四角 -> MVS 坐标：")
    for name, mx, my in mvs_corners:
        inside = 0 <= mx <= mvs_w and 0 <= my <= mvs_h
        flag = "inside" if inside else "OUTSIDE"
        print(f"{name}: mvs_x={mx:9.3f}, mvs_y={my:9.3f}  {flag}")

    xs = [p[1] for p in mvs_corners]
    ys = [p[2] for p in mvs_corners]
    print("-" * 72)
    print(f"FOV bbox in MVS: x=[{min(xs):.3f}, {max(xs):.3f}], y=[{min(ys):.3f}, {max(ys):.3f}]")
    print("=" * 72)

    segments = [
        ("top", mvs_corners[0], mvs_corners[1]),
        ("right", mvs_corners[1], mvs_corners[2]),
        ("bottom", mvs_corners[2], mvs_corners[3]),
        ("left", mvs_corners[3], mvs_corners[0]),
    ]

    if args.draw_diagonal:
        segments.extend([
            ("diag_LT_RB", mvs_corners[0], mvs_corners[2]),
            ("diag_RT_LB", mvs_corners[1], mvs_corners[3]),
        ])

    out_path = sync_root / f"overlay_bullet_point_{cam}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"写入目标: {out_path}")
    print(f"持续发送 {args.duration:.1f} 秒。确保 websocket_bullet.py 正在运行，UE5 连 ws://192.168.31.115:9876")
    print("你应该在 UE5 里看到 event-FOV 投影后的四边形，而不是整张 MVS 大框。")
    print("=" * 72)

    end_time = time.time() + args.duration
    loop_idx = 0

    with out_path.open("a", encoding="utf-8") as f:
        while time.time() < end_time:
            base_id = int(time.time() * 1000) % 100000000

            for idx, (seg_name, a, b) in enumerate(segments, start=1):
                _, x0, y0 = a
                _, x1, y1 = b

                bullet_id = base_id + idx
                sync_run_id = f"ue5_event_fov_{cam}_{seg_name}_{base_id}"

                write_segment(
                    f=f,
                    cam=cam,
                    sync_run_id=sync_run_id,
                    bullet_id=bullet_id,
                    segment_index=idx,
                    p0=(x0, y0),
                    p1=(x1, y1),
                    mvs_w=mvs_w,
                    mvs_h=mvs_h,
                    source="manual_event_fov_box_test",
                )
                time.sleep(0.02)

            loop_idx += 1
            print(f"已发送第 {loop_idx} 轮 event-FOV 框")
            time.sleep(args.loop_sleep)

    print("测试结束")


if __name__ == "__main__":
    main()
