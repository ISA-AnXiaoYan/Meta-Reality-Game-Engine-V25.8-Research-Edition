#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


VENDOR = os.environ.get("VENDOR", "1409").lower()
PRODUCT = os.environ.get("PRODUCT", "8e00").lower()
EXPECTED_COUNT = int(os.environ.get("EXPECTED_COUNT", "8") or "8")
INTERVAL = float(os.environ.get("INTERVAL", "0.5") or "0.5")
ACTIVE_THRESHOLD_MBPS = float(os.environ.get("ACTIVE_THRESHOLD_MBPS", "0.3") or "0.3")
DEFAULT_STATUS_JSON = os.environ.get("USB_STATUS_JSON", "/dev/shm/ids_usb_camera_throughput.json")
SYS_USB = Path("/sys/bus/usb/devices")
USBMON = Path("/sys/kernel/debug/usb/usbmon")
TOKEN_RE = re.compile(r"^([A-Z])([io]):(\d+):(\d+):(\d+)")


def path_exists(path: Path) -> Tuple[bool, str]:
    try:
        return path.exists(), ""
    except PermissionError as exc:
        return False, f"permission_denied: {path}: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {path}: {exc}"


def path_is_dir(path: Path) -> Tuple[bool, str]:
    try:
        return path.is_dir(), ""
    except PermissionError as exc:
        return False, f"permission_denied: {path}: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {path}: {exc}"


def read_text(path: Path, default: str = "") -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="ignore").strip()
        return value if value else default
    except Exception:
        return default


def run_quiet(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass


def setup_usbmon_if_root() -> None:
    if not (os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0):
        return
    run_quiet(["modprobe", "usbmon"])
    if Path("/sys/kernel/debug").is_dir() and not os.path.ismount("/sys/kernel/debug"):
        run_quiet(["mount", "-t", "debugfs", "none", "/sys/kernel/debug"])
    for _ in range(10):
        ok, _err = path_is_dir(USBMON)
        if ok:
            return
        time.sleep(0.1)


def find_cameras(vendor: str = VENDOR, product: str = PRODUCT) -> Dict[Tuple[int, int], Dict[str, Any]]:
    cameras: Dict[Tuple[int, int], Dict[str, Any]] = {}
    ok, _err = path_is_dir(SYS_USB)
    if not ok:
        return cameras
    try:
        devices = list(SYS_USB.iterdir())
    except Exception:
        return cameras
    for dev in devices:
        vid = read_text(dev / "idVendor").lower()
        pid = read_text(dev / "idProduct").lower()
        if not vid or not pid:
            continue
        if vid != str(vendor).lower() or pid != str(product).lower():
            continue
        try:
            bus = int(read_text(dev / "busnum", "0"))
            devnum = int(read_text(dev / "devnum", "0"))
        except Exception:
            continue
        serial = read_text(dev / "serial", "")
        cameras[(bus, devnum)] = {
            "usb_path": dev.name,
            "bus": bus,
            "dev": devnum,
            "serial": serial,
            "speed": read_text(dev / "speed", "N/A"),
            "product": read_text(dev / "product", "N/A"),
            "manufacturer": read_text(dev / "manufacturer", "N/A"),
            "rx_bytes": 0,
            "tx_bytes": 0,
            "urb_ok": 0,
            "urb_err": 0,
        }
    return cameras


def read_usbmon_bus(bus: int, interval: float) -> str:
    mon_path = USBMON / f"{int(bus)}u"
    ok, _err = path_exists(mon_path)
    if not ok:
        return ""
    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        return ""
    try:
        proc = subprocess.Popen(
            [timeout_bin, f"{max(0.1, float(interval)):.3f}", "cat", str(mon_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
        )
        out, _err = proc.communicate(timeout=max(1.0, float(interval) + 1.5))
        return out or ""
    except Exception:
        return ""


def parse_usbmon_text(text: str, cameras: Dict[Tuple[int, int], Dict[str, Any]]) -> None:
    for line in str(text or "").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        if parts[2] != "C":
            continue
        match = TOKEN_RE.match(parts[3])
        if not match:
            continue
        direction = match.group(2)
        try:
            bus = int(match.group(3))
            devnum = int(match.group(4))
            status = int(parts[4])
            length = int(parts[5])
        except Exception:
            continue
        cam = cameras.get((bus, devnum))
        if cam is None:
            continue
        if status != 0:
            cam["urb_err"] += 1
            continue
        if length <= 0:
            continue
        if direction == "i":
            cam["rx_bytes"] += length
        else:
            cam["tx_bytes"] += length
        cam["urb_ok"] += 1


def sample(interval: float, vendor: str = VENDOR, product: str = PRODUCT) -> Dict[str, Any]:
    cameras = find_cameras(vendor, product)
    buses = sorted({bus for bus, _dev in cameras.keys()})
    bus_outputs: Dict[int, str] = {}
    procs: List[Tuple[int, subprocess.Popen]] = []
    timeout_bin = shutil.which("timeout")
    errors: List[str] = []

    usbmon_ok, usbmon_err = path_is_dir(USBMON)
    if not usbmon_ok:
        if usbmon_err:
            errors.append(f"{usbmon_err}; run sudo bash ./setup_usbmon_permissions.sh")
        else:
            errors.append("usbmon path not found; run sudo bash ./setup_usbmon_permissions.sh")
    if timeout_bin is None:
        errors.append("timeout command not found")

    if usbmon_ok and timeout_bin:
        for bus in buses:
            mon_path = USBMON / f"{bus}u"
            mon_ok, mon_err = path_exists(mon_path)
            if not mon_ok:
                if mon_err:
                    errors.append(mon_err)
                continue
            try:
                proc = subprocess.Popen(
                    [timeout_bin, f"{max(0.1, float(interval)):.3f}", "cat", str(mon_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="ignore",
                )
                procs.append((bus, proc))
            except Exception as exc:
                errors.append(f"{mon_path}: {type(exc).__name__}: {exc}")
        for bus, proc in procs:
            try:
                out, err = proc.communicate(timeout=max(1.0, float(interval) + 1.5))
                bus_outputs[bus] = out or ""
                if proc.returncode not in (0, 124) and err:
                    errors.append(err.strip()[-200:])
            except Exception as exc:
                errors.append(f"usbmon bus {bus}: {type(exc).__name__}: {exc}")
        for text in bus_outputs.values():
            parse_usbmon_text(text, cameras)

    devices: Dict[str, Dict[str, Any]] = {}
    serials: Dict[str, Dict[str, Any]] = {}
    interval_safe = max(0.001, float(interval or INTERVAL))
    active_count = 0
    usb2_bad = 0
    usb3_ok = 0
    for key in sorted(cameras.keys(), key=lambda k: cameras[k]["usb_path"]):
        cam = cameras[key]
        rx_MBps = float(cam["rx_bytes"]) / interval_safe / 1_000_000.0
        tx_MBps = float(cam["tx_bytes"]) / interval_safe / 1_000_000.0
        total_MBps = rx_MBps + tx_MBps
        status = "ACTIVE" if total_MBps >= ACTIVE_THRESHOLD_MBPS else ("LOW" if total_MBps >= 0.1 else "IDLE")
        if status == "ACTIVE":
            active_count += 1
        speed = str(cam.get("speed", ""))
        if speed in ("5000", "10000", "20000"):
            usb3_ok += 1
        elif speed == "480":
            usb2_bad += 1
        rec = {
            "usb_path": cam["usb_path"],
            "bus": int(cam["bus"]),
            "dev": int(cam["dev"]),
            "serial": str(cam.get("serial", "")),
            "speed": speed,
            "product": str(cam.get("product", "")),
            "manufacturer": str(cam.get("manufacturer", "")),
            "rx_MBps": rx_MBps,
            "tx_MBps": tx_MBps,
            "total_MBps": total_MBps,
            "mbps": total_MBps,
            "urb_ok_s": float(cam["urb_ok"]) / interval_safe,
            "urb_err_s": float(cam["urb_err"]) / interval_safe,
            "data_status": status,
        }
        dev_key = f"{cam['bus']}:{cam['dev']}"
        devices[dev_key] = rec
        if rec["serial"]:
            serials[rec["serial"]] = rec

    return {
        "ok": len(errors) == 0,
        "timestamp_wall": time.time(),
        "interval_sec": interval_safe,
        "vendor": str(vendor).lower(),
        "product": str(product).lower(),
        "expected_count": int(EXPECTED_COUNT),
        "active_threshold_mbps": float(ACTIVE_THRESHOLD_MBPS),
        "detected_count": len(cameras),
        "active_count": int(active_count),
        "usb3_ok": int(usb3_ok),
        "usb2_bad": int(usb2_bad),
        "error": "; ".join(x for x in errors if x),
        "setup_hint": "sudo bash ./setup_usbmon_permissions.sh",
        "devices": devices,
        "serials": serials,
    }


def _filter_serials(payload: Dict[str, Any], serials: Iterable[str]) -> Dict[str, Any]:
    wanted = [str(s).strip() for s in serials if str(s).strip()]
    if not wanted:
        return payload
    all_serials = payload.get("serials", {})
    if not isinstance(all_serials, dict):
        all_serials = {}
    payload = dict(payload)
    payload["serials"] = {s: all_serials.get(s, {"mbps": 0.0, "data_status": "MISSING"}) for s in wanted}
    threshold = float(payload.get("active_threshold_mbps", ACTIVE_THRESHOLD_MBPS) or ACTIVE_THRESHOLD_MBPS)
    payload["active_count"] = sum(1 for rec in payload["serials"].values() if float(rec.get("mbps", rec.get("total_MBps", 0.0)) or 0.0) >= threshold)
    return payload


def write_status_json(payload: Dict[str, Any], path: str) -> None:
    raw = str(path or "").strip()
    if not raw:
        return
    try:
        p = Path(raw)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        pass


def _human(payload: Dict[str, Any]) -> str:
    lines = [
        "IDS Camera Actual USB Throughput Monitor",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"VID:PID: {payload.get('vendor', VENDOR)}:{payload.get('product', PRODUCT)}",
        f"Interval: {float(payload.get('interval_sec', INTERVAL)):.2f}s",
        "",
        f"{'LINK':<10} {'DATA':<8} {'USB_PATH':<12} {'BUS':<5} {'DEV':<5} {'NEG_SPEED':<10} {'RX_MB/s':>10} {'TX_MB/s':>10} {'URB_OK/s':>10} {'URB_ERR/s':>10}",
        "-" * 105,
    ]
    for rec in payload.get("devices", {}).values():
        speed = str(rec.get("speed", "N/A"))
        link = "USB3_LINK" if speed in ("5000", "10000", "20000") else ("USB2_BAD" if speed == "480" else "CHECK")
        lines.append(
            f"{link:<10} {str(rec.get('data_status','')):<8} {str(rec.get('usb_path','')):<12} "
            f"{int(rec.get('bus',0)):<5} {int(rec.get('dev',0)):<5} {(speed + 'M'):<10} "
            f"{float(rec.get('rx_MBps',0.0)):>10.2f} {float(rec.get('tx_MBps',0.0)):>10.2f} "
            f"{float(rec.get('urb_ok_s',0.0)):>10.0f} {float(rec.get('urb_err_s',0.0)):>10.0f}"
        )
    lines += [
        "",
        "Summary:",
        f"  Expected cameras : {payload.get('expected_count', EXPECTED_COUNT)}",
        f"  Detected cameras : {payload.get('detected_count', 0)}",
        f"  USB3 link OK     : {payload.get('usb3_ok', 0)}",
        f"  USB2 downgraded  : {payload.get('usb2_bad', 0)}",
        f"  Active transfer  : {payload.get('active_count', 0)}",
    ]
    if payload.get("error"):
        lines += ["", f"ERROR: {payload.get('error')}"]
    lines += ["", "Press Ctrl+C to stop."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor IDS event camera USB throughput.")
    ap.add_argument("--serials", default="", help="Optional comma-separated serials for JSON filtering.")
    ap.add_argument("--interval", type=float, default=INTERVAL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--vendor", default=VENDOR)
    ap.add_argument("--product", default=PRODUCT)
    ap.add_argument("--status-json", default=DEFAULT_STATUS_JSON, help="Write latest JSON payload here for non-root launch diagnostics.")
    args = ap.parse_args()

    if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0 and not args.json:
        print("ERROR: please run with sudo.")
        print("Example:")
        print("  sudo INTERVAL=0.5 ./monitor_camera_usb_throughput.py")
        return 1

    setup_usbmon_if_root()
    serial_filter = [x.strip() for x in str(args.serials or "").split(",") if x.strip()]
    while True:
        payload = sample(float(args.interval or INTERVAL), args.vendor, args.product)
        payload = _filter_serials(payload, serial_filter)
        write_status_json(payload, str(args.status_json or ""))
        if args.json:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            print("\033[2J\033[H" + _human(payload), flush=True)
        if args.once:
            return 0 if payload.get("detected_count", 0) > 0 else 1
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
