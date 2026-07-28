#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reset_event_usb_by_serial.py

按事件相机序列号对 USB 设备做软件重启：
    echo 0 > /sys/bus/usb/devices/<dev>/authorized
    echo 1 > /sys/bus/usb/devices/<dev>/authorized

用途：
- 避免频繁物理拔插 USB。
- 当 Metavision 事件相机提示 occupied / busy / device already opened 时，
  可先停掉程序，再对对应序列号做 USB soft reset。

默认相机映射：
    A: 4110040581
    B: 4110046946
    C: 4110046952
    D: 4110046954
    E: 4110046945
    F: 4110046947

推荐用法：
    sudo python3 tools/reset_event_usb_by_serial.py --all

只重启某几台：
    sudo python3 tools/reset_event_usb_by_serial.py --cams E,F

先预览不实际执行：
    python3 tools/reset_event_usb_by_serial.py --all --dry-run

同时先杀掉项目残留进程：
    sudo python3 tools/reset_event_usb_by_serial.py --all --kill-first
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CAMERA_SERIALS: Dict[str, str] = {
    "A": "4110040581",
    "B": "4110046946",
    "C": "4110046952",
    "D": "4110046954",
    "E": "4110046945",
    "F": "4110046947",
}

KILL_PATTERNS = [
    "launch_fusion_system.py",
    "multi_camera_launcher.py",
    "metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py",
    "hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py",
    "fusion_renderer_gl.py",
    "hit_judge_server.py",
    "ue5_human_stream_bridge.py",
    "ue5_overlay_jsonl_ws_forwarder.py",
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def find_usb_devices_by_serial() -> Dict[str, List[Path]]:
    """
    返回 serial -> [usb_device_path, ...]
    只扫描 /sys/bus/usb/devices 下带 serial 文件的 USB device。
    """
    root = Path("/sys/bus/usb/devices")
    found: Dict[str, List[Path]] = {}
    if not root.exists():
        return found

    for dev in sorted(root.iterdir(), key=lambda p: p.name):
        serial_file = dev / "serial"
        authorized_file = dev / "authorized"
        if not serial_file.exists():
            continue
        # 没有 authorized 的通常不是可直接 reset 的 USB device，跳过
        if not authorized_file.exists():
            continue

        serial = read_text(serial_file)
        if not serial:
            continue
        found.setdefault(serial, []).append(dev)

    return found


def describe_device(dev: Path) -> str:
    fields = {
        "dev": dev.name,
        "serial": read_text(dev / "serial"),
        "vendor": read_text(dev / "idVendor"),
        "product_id": read_text(dev / "idProduct"),
        "manufacturer": read_text(dev / "manufacturer"),
        "product": read_text(dev / "product"),
        "busnum": read_text(dev / "busnum"),
        "devnum": read_text(dev / "devnum"),
    }
    return (
        f"{fields['dev']} serial={fields['serial']} "
        f"vid:pid={fields['vendor']}:{fields['product_id']} "
        f"bus={fields['busnum']} devnum={fields['devnum']} "
        f"manufacturer={fields['manufacturer']!r} product={fields['product']!r}"
    )


def parse_cams(cams: Optional[str], all_flag: bool) -> List[str]:
    if all_flag or not cams:
        return list(CAMERA_SERIALS.keys())

    out: List[str] = []
    for item in cams.split(","):
        cam = item.strip().upper()
        if not cam:
            continue
        if cam not in CAMERA_SERIALS:
            raise SystemExit(f"[ERROR] unknown cam alias: {cam}. valid={','.join(CAMERA_SERIALS)}")
        out.append(cam)

    # 去重但保序
    unique: List[str] = []
    seen = set()
    for cam in out:
        if cam not in seen:
            unique.append(cam)
            seen.add(cam)
    return unique


def kill_stale_processes(dry_run: bool = False) -> None:
    print("[kill] try graceful terminate stale project processes...")
    for pat in KILL_PATTERNS:
        cmd = ["pkill", "-TERM", "-f", pat]
        print("[kill]", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=False)

    if not dry_run:
        time.sleep(2.0)

    print("[kill] force kill remaining stale project processes...")
    for pat in KILL_PATTERNS:
        cmd = ["pkill", "-9", "-f", pat]
        print("[kill]", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=False)


def write_authorized(dev: Path, value: str, dry_run: bool) -> None:
    authorized = dev / "authorized"
    if dry_run:
        print(f"[dry-run] write {value!r} -> {authorized}")
        return

    try:
        authorized.write_text(value, encoding="utf-8")
    except PermissionError:
        raise SystemExit(
            f"[ERROR] permission denied: {authorized}\n"
            f"请用 sudo 运行：sudo python3 tools/reset_event_usb_by_serial.py --all"
        )
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] authorized file disappeared: {authorized}")


def reset_usb_device(dev: Path, off_sec: float, on_sec: float, dry_run: bool) -> None:
    print(f"[USB RESET] {describe_device(dev)}")
    print(f"[USB RESET] disable authorized=0")
    write_authorized(dev, "0", dry_run=dry_run)
    if not dry_run:
        time.sleep(off_sec)

    print(f"[USB RESET] enable authorized=1")
    write_authorized(dev, "1", dry_run=dry_run)
    if not dry_run:
        time.sleep(on_sec)


def run_metavision_list() -> None:
    exe = "metavision_hal_test"
    print("\n[check] try: metavision_hal_test --list")
    try:
        subprocess.run([exe, "--list"], check=False)
    except FileNotFoundError:
        print("[check][WARN] metavision_hal_test not found in PATH")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Soft reset event-camera USB devices by serial number."
    )
    parser.add_argument(
        "--cams",
        default="",
        help="camera aliases to reset, e.g. A,B,C or E,F. Default: all A-F.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="reset all configured cameras A-F.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print what would be reset.",
    )
    parser.add_argument(
        "--kill-first",
        action="store_true",
        help="pkill project processes before USB reset.",
    )
    parser.add_argument(
        "--off-sec",
        type=float,
        default=1.2,
        help="seconds to wait after authorized=0. Default: 1.2.",
    )
    parser.add_argument(
        "--on-sec",
        type=float,
        default=2.5,
        help="seconds to wait after authorized=1. Default: 2.5.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run metavision_hal_test --list after reset.",
    )
    args = parser.parse_args()

    cams = parse_cams(args.cams, args.all)
    serials = [(cam, CAMERA_SERIALS[cam]) for cam in cams]

    print("[config] cameras:")
    for cam, serial in serials:
        print(f"  {cam}: {serial}")

    if args.kill_first:
        kill_stale_processes(dry_run=args.dry_run)

    found = find_usb_devices_by_serial()

    print("\n[scan] USB devices with serial:")
    if not found:
        print("  [WARN] no USB serial devices found under /sys/bus/usb/devices")
    else:
        for serial, devs in sorted(found.items()):
            for dev in devs:
                mark = ""
                for cam, cam_serial in serials:
                    if cam_serial == serial:
                        mark = f"  <-- {cam}"
                        break
                print(" ", describe_device(dev) + mark)

    missing: List[Tuple[str, str]] = []
    reset_count = 0

    print("\n[reset] start")
    for cam, serial in serials:
        devs = found.get(serial, [])
        if not devs:
            print(f"[WARN] {cam} serial={serial} not found in /sys/bus/usb/devices")
            missing.append((cam, serial))
            continue

        if len(devs) > 1:
            print(f"[WARN] {cam} serial={serial} matched multiple devices; reset all matches.")

        for dev in devs:
            reset_usb_device(dev, off_sec=args.off_sec, on_sec=args.on_sec, dry_run=args.dry_run)
            reset_count += 1

    print(f"\n[done] reset_count={reset_count}, missing={len(missing)}")
    if missing:
        print("[missing]")
        for cam, serial in missing:
            print(f"  {cam}: {serial}")

    if args.check and not args.dry_run:
        # 给 udev/SDK 一点时间
        time.sleep(1.5)
        run_metavision_list()

    if args.dry_run:
        print("\n[dry-run] no device was modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
