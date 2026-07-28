#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usb_recover_v2.py

用途：
1. 列出当前 USB 设备、root hub、xHCI 控制器。
2. 优先尝试按 USB device authorized=0/1 reset。
3. 如果事件相机序列号没有出现在 /sys/bus/usb/devices/*/serial，
   则使用 xHCI host controller unbind/bind 做软件层面的“USB 控制器重启”。

注意：
- 重启 xHCI 控制器会让该控制器下所有 USB 设备重新枚举。
- 如果键盘/鼠标/网卡也挂在该控制器上，会短暂断开。
- 建议在本机终端或远程 SSH 中谨慎操作。
- 先运行 --list 看清楚控制器，再 reset。

根据你当前机器扫描到的控制器：
    0000:41:00.4  -> usb3 / usb4
    0000:6a:00.0  -> usb5 / usb6
    0000:6e:00.0  -> usb7 / usb8
    0000:e1:00.4  -> usb1 / usb2

常用命令：

只列出：
    python3 tools/usb_recover_v2.py --list

重启某一个 xHCI 控制器：
    sudo python3 tools/usb_recover_v2.py --reset-controller 0000:6a:00.0

依次重启多个控制器：
    sudo python3 tools/usb_recover_v2.py --reset-controller 0000:41:00.4,0000:6a:00.0,0000:6e:00.0

重启从扫描结果里看到的所有非 MediaTek root hub 对应控制器：
    sudo python3 tools/usb_recover_v2.py --reset-known-controllers

如果你还想先杀项目残留进程：
    sudo python3 tools/usb_recover_v2.py --kill-first --reset-known-controllers
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EVENT_SERIALS = {
    "A": "4110040581",
    "B": "4110046946",
    "C": "4110046952",
    "D": "4110046954",
    "E": "4110046945",
    "F": "4110046947",
}

# 这是你当前日志里扫描到的 xHCI controller PCI 地址。
# 如果以后硬件插槽变化，可用 --list 重新查看。
KNOWN_XHCI_CONTROLLERS = [
    "0000:41:00.4",
    "0000:6a:00.0",
    "0000:6e:00.0",
    "0000:e1:00.4",
]

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


def write_text_root(path: Path, value: str) -> None:
    try:
        path.write_text(value, encoding="utf-8")
    except PermissionError:
        raise SystemExit(f"[ERROR] permission denied: {path}\n请用 sudo 运行。")
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] path not found: {path}")


def run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def kill_first() -> None:
    print("[kill] graceful terminate project processes")
    for pat in KILL_PATTERNS:
        subprocess.run(["pkill", "-TERM", "-f", pat], check=False)
    time.sleep(2.0)

    print("[kill] force kill remaining project processes")
    for pat in KILL_PATTERNS:
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(0.5)


def usb_device_records() -> List[Dict[str, str]]:
    root = Path("/sys/bus/usb/devices")
    rows: List[Dict[str, str]] = []
    if not root.exists():
        return rows

    for dev in sorted(root.iterdir(), key=lambda p: p.name):
        if not dev.is_dir():
            continue

        row = {
            "name": dev.name,
            "path": str(dev),
            "serial": read_text(dev / "serial"),
            "idVendor": read_text(dev / "idVendor"),
            "idProduct": read_text(dev / "idProduct"),
            "manufacturer": read_text(dev / "manufacturer"),
            "product": read_text(dev / "product"),
            "busnum": read_text(dev / "busnum"),
            "devnum": read_text(dev / "devnum"),
            "devpath": read_text(dev / "devpath"),
            "speed": read_text(dev / "speed"),
            "authorized_exists": "1" if (dev / "authorized").exists() else "0",
            "driver": "",
            "pci_controller": "",
        }

        driver_link = dev / "driver"
        try:
            if driver_link.exists() or driver_link.is_symlink():
                row["driver"] = os.path.basename(os.path.realpath(driver_link))
        except Exception:
            pass

        # root hub usbX 的 device 链接通常指向 PCI controller
        try:
            device_real = os.path.realpath(dev / "device")
            # example: /sys/devices/pci0000:40/0000:40:03.1/0000:41:00.4
            base = os.path.basename(device_real)
            if base.startswith("0000:"):
                row["pci_controller"] = base
        except Exception:
            pass

        # 对普通 USB 子设备，向上找最近的 pci controller
        if not row["pci_controller"]:
            p = dev.resolve()
            for parent in [p] + list(p.parents):
                b = parent.name
                if b.startswith("0000:") and len(b) >= 12:
                    row["pci_controller"] = b
                    break

        rows.append(row)

    return rows


def print_list() -> None:
    rows = usb_device_records()
    print("\n[USB devices]")
    if not rows:
        print("  no devices found under /sys/bus/usb/devices")
    for r in rows:
        if not (r["idVendor"] or r["idProduct"] or r["serial"] or r["product"] or r["name"].startswith("usb")):
            continue
        mark = ""
        for cam, ser in EVENT_SERIALS.items():
            if r["serial"] == ser:
                mark = f"  <-- EVENT {cam}"
                break
        print(
            f"  {r['name']:>8} bus={r['busnum'] or '-'} dev={r['devnum'] or '-'} "
            f"vid:pid={r['idVendor'] or '----'}:{r['idProduct'] or '----'} "
            f"serial={r['serial'] or '-'} product={r['product'] or '-'} "
            f"mfg={r['manufacturer'] or '-'} driver={r['driver'] or '-'} "
            f"ctrl={r['pci_controller'] or '-'} authorized={r['authorized_exists']}{mark}"
        )

    print("\n[xHCI root hubs / controllers]")
    seen = set()
    for r in rows:
        if not r["name"].startswith("usb"):
            continue
        ctrl = r["pci_controller"]
        if not ctrl:
            continue
        key = (ctrl, r["name"])
        if key in seen:
            continue
        seen.add(key)
        print(
            f"  {r['name']:>5} ctrl={ctrl} bus={r['busnum'] or '-'} "
            f"vid:pid={r['idVendor']}:{r['idProduct']} product={r['product']}"
        )

    print("\n[known controllers from your last scan]")
    for ctrl in KNOWN_XHCI_CONTROLLERS:
        print(f"  {ctrl}")


def reset_authorized_device(sysfs_path: str, off_sec: float, on_sec: float, dry_run: bool) -> None:
    dev = Path(sysfs_path)
    auth = dev / "authorized"
    if not auth.exists():
        print(f"[WARN] no authorized file: {auth}")
        return

    print(f"[USB device reset] {dev}")
    if dry_run:
        print(f"[dry-run] write 0 -> {auth}")
        print(f"[dry-run] write 1 -> {auth}")
        return

    write_text_root(auth, "0")
    time.sleep(off_sec)
    write_text_root(auth, "1")
    time.sleep(on_sec)


def reset_by_visible_serials(off_sec: float, on_sec: float, dry_run: bool) -> int:
    rows = usb_device_records()
    count = 0
    wanted = set(EVENT_SERIALS.values())
    for r in rows:
        if r["serial"] in wanted and r["authorized_exists"] == "1":
            reset_authorized_device(r["path"], off_sec=off_sec, on_sec=on_sec, dry_run=dry_run)
            count += 1
    return count


def reset_xhci_controller(ctrl: str, off_sec: float, on_sec: float, dry_run: bool) -> None:
    ctrl = ctrl.strip()
    if not ctrl:
        return

    pci_dev = Path("/sys/bus/pci/devices") / ctrl
    if not pci_dev.exists():
        print(f"[WARN] PCI controller not found: {ctrl}")
        return

    # xhci_hcd driver path
    unbind = Path("/sys/bus/pci/drivers/xhci_hcd/unbind")
    bind = Path("/sys/bus/pci/drivers/xhci_hcd/bind")
    if not unbind.exists() or not bind.exists():
        raise SystemExit("[ERROR] xhci_hcd bind/unbind path not found.")

    print(f"\n[xHCI RESET] controller={ctrl}")
    print("[xHCI RESET] this will reset all USB devices under this controller.")
    if dry_run:
        print(f"[dry-run] write {ctrl} -> {unbind}")
        print(f"[dry-run] write {ctrl} -> {bind}")
        return

    write_text_root(unbind, ctrl)
    time.sleep(off_sec)
    write_text_root(bind, ctrl)
    time.sleep(on_sec)


def run_metavision_list() -> None:
    print("\n[check] metavision_hal_test --list")
    try:
        subprocess.run(["metavision_hal_test", "--list"], check=False)
    except FileNotFoundError:
        print("[WARN] metavision_hal_test not found in PATH")


def run_lsusb() -> None:
    print("\n[check] lsusb")
    try:
        subprocess.run(["lsusb"], check=False)
    except FileNotFoundError:
        print("[WARN] lsusb not found")

    print("\n[check] lsusb -t")
    try:
        subprocess.run(["lsusb", "-t"], check=False)
    except FileNotFoundError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list USB devices and controllers.")
    ap.add_argument("--kill-first", action="store_true", help="kill project processes before reset.")
    ap.add_argument("--reset-visible-serials", action="store_true", help="reset event cameras only if their serial appears in sysfs.")
    ap.add_argument("--reset-controller", default="", help="comma-separated PCI xHCI controllers, e.g. 0000:6a:00.0,0000:6e:00.0")
    ap.add_argument("--reset-known-controllers", action="store_true", help="reset known xHCI controllers from your last scan.")
    ap.add_argument("--off-sec", type=float, default=2.0)
    ap.add_argument("--on-sec", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="run lsusb and metavision_hal_test after reset.")
    args = ap.parse_args()

    if args.list:
        print_list()
        return 0

    if args.kill_first:
        kill_first()

    did = False

    if args.reset_visible_serials:
        print("[reset] try visible serial devices")
        count = reset_by_visible_serials(args.off_sec, args.on_sec, args.dry_run)
        print(f"[reset] visible serial reset count={count}")
        did = True

    ctrls: List[str] = []
    if args.reset_known_controllers:
        ctrls.extend(KNOWN_XHCI_CONTROLLERS)
    if args.reset_controller.strip():
        ctrls.extend([x.strip() for x in args.reset_controller.split(",") if x.strip()])

    # 去重但保序
    unique_ctrls: List[str] = []
    seen = set()
    for c in ctrls:
        if c not in seen:
            unique_ctrls.append(c)
            seen.add(c)

    for ctrl in unique_ctrls:
        reset_xhci_controller(ctrl, args.off_sec, args.on_sec, args.dry_run)
        did = True

    if not did:
        print("[INFO] no reset action specified. Use --list or --reset-controller/--reset-known-controllers.")
        print("Examples:")
        print("  python3 tools/usb_recover_v2.py --list")
        print("  sudo python3 tools/usb_recover_v2.py --reset-controller 0000:6a:00.0 --check")
        return 0

    if args.check and not args.dry_run:
        time.sleep(2.0)
        run_lsusb()
        run_metavision_list()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
