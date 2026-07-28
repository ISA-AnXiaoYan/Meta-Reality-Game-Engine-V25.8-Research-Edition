#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-camera Hik/MVS worker driven by trigger_coordinator tick messages.

Threaded version for 40 fps / 25 ms sync:
  - trigger thread: receives future ticks and calls TriggerSoftware on time.
  - read thread: blocks on source.read(), then publishes /dev/shm + audit.

Only this worker internals are changed.  The published metadata fields are kept
compatible with the old single-threaded split-worker implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

try:
    from preview_gpu_shm import RawPreviewFrameWriter, RAW_PIXEL_BAYER8
except Exception:
    RawPreviewFrameWriter = None
    RAW_PIXEL_BAYER8 = 1

# Import existing camera/config/shm implementations from the main MVS/YOLO file.
from hik_rgb_yolo_seg_multicam_same_model_stableid_v5 import (
    HikFullViewSource,
    MvsFrameAuditWriter,
    MvsRawRecorderManager,
    MvsSharedFramePublisher,
    _apply_sections_to_args,
    _as_bool,
    _build_route_args_direct,
    _deep_update,
    _make_base_default_args,
    load_direct_config,
    now_us,
)


def _find_route(cfg: Dict[str, Any], cam_id: str) -> Dict[str, Any]:
    for r in cfg.get("routes", []) or []:
        if not isinstance(r, dict) or not bool(r.get("enabled", True)):
            continue
        cid = str(r.get("id", r.get("name", "")) or "").strip()
        if cid == str(cam_id):
            return r
    raise RuntimeError(f"route {cam_id!r} not found in config")


def _build_args_for_route(cfg: Dict[str, Any], cam_id: str) -> tuple[argparse.Namespace, Dict[str, Any], Dict[str, Any]]:
    common_cfg = cfg.get("common", {}) or {}
    hik_defaults = cfg.get("hik_defaults", {}) or {}
    sync_cfg = copy.deepcopy(cfg.get("sync", {}) or {})
    route_cfg = _find_route(cfg, cam_id)
    default_args = _make_base_default_args()
    args = _build_route_args_direct(default_args, common_cfg, route_cfg, hik_defaults)
    if isinstance(route_cfg.get("sync"), dict):
        _deep_update(sync_cfg, route_cfg["sync"])
    for key in (
        "exposure_us", "gain", "exposure_auto_off", "gain_auto_off",
        "exposure_auto", "auto_exposure_time_lower_limit_us", "auto_exposure_time_upper_limit_us",
        "auto_exposure_lower_us", "auto_exposure_upper_us",
        "gain_auto", "auto_gain_lower_limit", "auto_gain_upper_limit",
        "auto_gain_lower", "auto_gain_upper",
    ):
        if key in route_cfg and route_cfg.get(key) is not None:
            sync_cfg[key] = copy.deepcopy(route_cfg.get(key))
    # In split-worker mode this worker owns the camera and is externally clocked.
    # Keep TriggerMode=On/TriggerSource=Software, but do not use the old in-process
    # central trigger group.
    sync_cfg["enable"] = True
    sync_cfg["mvs_trigger_mode"] = str(sync_cfg.get("mvs_trigger_mode", "On") or "On")
    sync_cfg["trigger_source"] = str(sync_cfg.get("trigger_source", "Software") or "Software")
    setattr(args, "sync_cfg", sync_cfg)
    return args, sync_cfg, route_cfg


def _sock_dir(sync_cfg: Dict[str, Any], override: str = "") -> Path:
    root = str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
    d = str(override or sync_cfg.get("mvs_trigger_ipc_dir", "{root}/trigger_ipc") or "{root}/trigger_ipc")
    d = d.format(root=root)
    return Path(d).expanduser().resolve()


def _socket_path(sock_dir: Path, cam: str, template: str) -> Path:
    return sock_dir / template.format(camera=str(cam), cam=str(cam), id=str(cam))


def _sleep_until_ns(target_ns: int, spin_us: float = 300.0) -> None:
    spin_ns = int(max(0.0, spin_us) * 1000.0)
    while True:
        now = time.monotonic_ns()
        remain = int(target_ns) - now
        if remain <= 0:
            return
        if remain > spin_ns + 1_000_000:
            time.sleep(min(0.002, (remain - spin_ns) / 1e9))
        elif remain > spin_ns:
            time.sleep((remain - spin_ns) / 1e9)
        else:
            # short spin for sub-ms trigger jitter
            while time.monotonic_ns() < int(target_ns):
                pass
            return


def _open_source(cam_id: str, args: argparse.Namespace) -> HikFullViewSource:
    if str(args.source).lower() != "hik":
        raise RuntimeError(f"split MVS worker only supports hik source, route {cam_id} source={args.source}")
    return HikFullViewSource(
        serial=args.serial,
        mvs_sdk_path=args.mvs_sdk_path,
        timeout_ms=int(getattr(args, "hik_timeout_ms", 1000) or 1000),
        width=int(getattr(args, "capture_width", 0) or 0),
        height=int(getattr(args, "capture_height", 0) or 0),
        fps=float(getattr(args, "capture_fps", 0.0) or 0.0),
        force_full_sensor=bool(getattr(args, "force_full_sensor", True)),
        prefer_bgr8=bool(getattr(args, "prefer_bgr8", False)),
        sync_cfg=getattr(args, "sync_cfg", {}) or {},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="One Hik/MVS camera split worker driven by trigger coordinator")
    ap.add_argument("--config", required=True, help="MVS runtime JSON")
    ap.add_argument("--cam", required=True, help="Camera id, e.g. A")
    ap.add_argument("--socket-dir", default="")
    ap.add_argument("--socket-template", default="trigger_{camera}.sock")
    ap.add_argument("--late-drop-ms", type=float, default=-1.0, help="Drop a tick if received this late. Default sync.external_trigger_late_drop_ms or 8ms")
    ap.add_argument("--spin-us", type=float, default=-1.0, help="Busy-spin window before fire_time. Default 300us")
    ap.add_argument("--recv-timeout-ms", type=float, default=500.0)
    ap.add_argument("--print-every", type=int, default=40)
    ap.add_argument("--raw-preview-shm-enable", action="store_true", help="Publish raw Bayer8 latest frame for PyOpenGL/CUDA preview renderer")
    ap.add_argument("--raw-preview-shm-template", default="mvs_raw_latest_{camera}", help="Raw preview shm name/path template")
    ap.add_argument("--raw-preview-slot-count", type=int, default=3)
    ap.add_argument("--raw-preview-bayer-pattern", default="", help="Override raw Bayer pattern for preview; default source/config BAYER_PATTERN")
    args_cli = ap.parse_args()

    cam_id = str(args_cli.cam).strip()
    cfg = load_direct_config(args_cli.config)
    args, sync_cfg, route_cfg = _build_args_for_route(cfg, cam_id)
    sock_dir = _sock_dir(sync_cfg, args_cli.socket_dir)
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = _socket_path(sock_dir, cam_id, args_cli.socket_template)
    try:
        sock_path.unlink()
    except FileNotFoundError:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(sock_path))
    sock.settimeout(max(0.05, float(args_cli.recv_timeout_ms) / 1000.0))

    late_drop_ms = float(args_cli.late_drop_ms if args_cli.late_drop_ms >= 0 else sync_cfg.get("external_trigger_late_drop_ms", 8.0) or 8.0)
    spin_us = float(args_cli.spin_us if args_cli.spin_us >= 0 else sync_cfg.get("external_trigger_spin_us", 300.0) or 300.0)
    pending_max = int(sync_cfg.get("external_trigger_pending_queue_max", 512) or 512)
    pending_max = max(16, pending_max)
    publisher = MvsSharedFramePublisher(sync_cfg, [cam_id])
    audit = MvsFrameAuditWriter(sync_cfg, [cam_id])
    mvs_recorder: Optional[MvsRawRecorderManager] = None
    record_cfg = copy.deepcopy(cfg.get("one_key_record", {}) or {})
    if isinstance(record_cfg, dict) and _as_bool(record_cfg.get("enable", False), False):
        sync_root = Path(str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")).expanduser()
        if not sync_root.is_absolute():
            sync_root = Path.cwd() / sync_root
        status_template = str(record_cfg.get("split_status_file_template") or str(sync_root / "record_status_{camera}.json"))
        try:
            record_cfg["status_file"] = status_template.format(
                root=str(sync_root.resolve()),
                camera=cam_id,
                cam=cam_id,
                id=cam_id,
            )
        except Exception:
            record_cfg["status_file"] = str((sync_root / f"record_status_{cam_id}.json").resolve())
        mvs_recorder = MvsRawRecorderManager(record_cfg, sync_cfg, [cam_id], {cam_id: args})
        mvs_recorder.start()
    raw_preview_enabled = bool(args_cli.raw_preview_shm_enable)
    raw_preview_writer = None
    raw_preview_shape = None
    raw_preview_name = str(args_cli.raw_preview_shm_template or "mvs_raw_latest_{camera}").format(camera=cam_id, cam=cam_id, id=cam_id)
    if raw_preview_enabled and RawPreviewFrameWriter is None:
        print(f"[mvs_worker][{cam_id}][WARN] preview_gpu_shm.RawPreviewFrameWriter unavailable; raw preview disabled", flush=True)
        raw_preview_enabled = False

    def _ensure_raw_preview_writer(raw_frame: np.ndarray, raw_meta: Dict[str, Any]):
        nonlocal raw_preview_writer, raw_preview_shape
        if not raw_preview_enabled or raw_frame is None:
            return None
        h, w = raw_frame.shape[:2]
        stride = int(raw_meta.get("stride", w) or w)
        shape = (int(h), int(w), int(stride))
        if raw_preview_writer is not None and raw_preview_shape == shape:
            return raw_preview_writer
        try:
            if raw_preview_writer is not None:
                try:
                    raw_preview_writer.close(unlink=True)
                except Exception:
                    pass
            pattern = str(args_cli.raw_preview_bayer_pattern or raw_meta.get("bayer_pattern") or getattr(source, "bayer_pattern", "BG") or "BG").upper()
            raw_preview_writer = RawPreviewFrameWriter(
                raw_preview_name, width=int(w), height=int(h), stride=int(stride),
                pixel_format=RAW_PIXEL_BAYER8, bayer_pattern=pattern,
                slot_count=int(args_cli.raw_preview_slot_count), create=True, flush=False,
            )
            raw_preview_shape = shape
            print(f"[mvs_worker][{cam_id}] raw preview shm enabled: {raw_preview_name} shape={h}x{w} stride={stride} pattern={pattern}", flush=True)
            return raw_preview_writer
        except Exception as exc:
            print(f"[mvs_worker][{cam_id}][WARN] create raw preview writer failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    stop_evt = threading.Event()

    def _stop(signum=None, frame=None):
        stop_evt.set()

    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    print(
        f"[mvs_worker][{cam_id}] socket={sock_path} late_drop_ms={late_drop_ms:.3f} spin_us={spin_us:.1f} "
        f"pending_max={pending_max} serial={getattr(args,'serial','')} "
        f"capture={getattr(args,'capture_width',0)}x{getattr(args,'capture_height',0)}@{getattr(args,'capture_fps',0)}",
        flush=True,
    )

    source: Optional[HikFullViewSource] = None

    # Metadata for triggers that have already been fired and are waiting for the
    # matching frame.  Order matters: Hik frames should arrive in trigger order.
    pending_meta: Deque[Dict[str, Any]] = deque()
    pending_lock = threading.Lock()

    stats_lock = threading.Lock()
    stats: Dict[str, Any] = {
        "trigger_ok_total": 0,
        "trigger_ok_window": 0,
        "drop_count": 0,
        "trigger_fail_count": 0,
        "timeout_count": 0,
        "orphan_frame_count": 0,
        "meta_overflow_count": 0,
        "read_ok_total": 0,
        "read_ok_window": 0,
        "frame_count": 0,
        "last_tick_id": -1,
        "last_trigger_delta_us": 0.0,
    }

    def _inc_stat(key: str, delta: int = 1) -> int:
        with stats_lock:
            stats[key] = int(stats.get(key, 0) or 0) + int(delta)
            return int(stats[key])

    def _set_stat(key: str, value: Any) -> None:
        with stats_lock:
            stats[key] = value

    def _snapshot_stats() -> Dict[str, Any]:
        with stats_lock:
            return dict(stats)

    def _push_trigger_meta(meta: Dict[str, Any]) -> None:
        overflow = 0
        with pending_lock:
            while len(pending_meta) >= pending_max:
                pending_meta.popleft()
                overflow += 1
            pending_meta.append(meta)
        if overflow > 0:
            total = _inc_stat("meta_overflow_count", overflow)
            if total <= 20:
                print(
                    f"[mvs_worker][{cam_id}][WARN] pending trigger metadata overflow, "
                    f"dropped_old={overflow} total={total}",
                    flush=True,
                )

    def _pop_trigger_meta() -> Optional[Dict[str, Any]]:
        with pending_lock:
            if not pending_meta:
                return None
            return pending_meta.popleft()

    def _pending_len() -> int:
        with pending_lock:
            return len(pending_meta)

    def trigger_loop() -> None:
        assert source is not None
        last_tick_id = -1
        while not stop_evt.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                if not stop_evt.is_set():
                    print(f"[mvs_worker][{cam_id}][WARN] trigger socket closed", flush=True)
                break
            except Exception as e:
                if not stop_evt.is_set():
                    print(f"[mvs_worker][{cam_id}][WARN] recv failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.01)
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if msg.get("type") != "mvs_trigger_tick":
                continue

            tick_id = int(msg.get("tick_id", 0) or 0)
            fire_ns = int(msg.get("fire_time_ns", 0) or 0)
            if tick_id <= last_tick_id:
                continue
            last_tick_id = tick_id
            _set_stat("last_tick_id", tick_id)

            now_ns = time.monotonic_ns()
            late_ms = (now_ns - fire_ns) / 1e6
            if fire_ns <= 0 or late_ms > late_drop_ms:
                drop_total = _inc_stat("drop_count", 1)
                if drop_total <= 20:
                    print(f"[mvs_worker][{cam_id}][WARN] drop late tick={tick_id} late_ms={late_ms:.3f}", flush=True)
                continue

            _sleep_until_ns(fire_ns, spin_us=spin_us)
            trigger_call_ns = time.monotonic_ns()
            trigger_exception = False
            try:
                cmd_ok, cmd_wall_us, cmd_done_wall_us, cmd_ret = source.software_trigger()
            except Exception as e:
                trigger_exception = True
                cmd_ok, cmd_wall_us, cmd_done_wall_us, cmd_ret = False, now_us(), now_us(), -1
                fail_total = _inc_stat("trigger_fail_count", 1)
                if fail_total <= 20:
                    print(f"[mvs_worker][{cam_id}][WARN] TriggerSoftware exception tick={tick_id}: {type(e).__name__}: {e}", flush=True)
            trigger_done_ns = time.monotonic_ns()
            if (not cmd_ok) and (not trigger_exception):
                _inc_stat("trigger_fail_count", 1)

            delta_us = (trigger_call_ns - fire_ns) / 1000.0
            _set_stat("last_trigger_delta_us", round(delta_us, 3))
            _inc_stat("trigger_ok_total", 1)
            _inc_stat("trigger_ok_window", 1)

            _push_trigger_meta({
                "tick_id": int(tick_id),
                "fire_ns": int(fire_ns),
                "fire_wall_us_est": int(msg.get("fire_wall_us_est", 0) or 0),
                "period_ns": int(msg.get("period_ns", 0) or 0),
                "send_time_ns": int(msg.get("send_time_ns", 0) or 0),
                "send_wall_us": int(msg.get("send_wall_us", 0) or 0),
                "fps": float(msg.get("fps", 0.0) or 0.0),
                "trigger_call_ns": int(trigger_call_ns),
                "trigger_done_ns": int(trigger_done_ns),
                "cmd_ok": bool(cmd_ok),
                "cmd_wall_us": int(cmd_wall_us),
                "cmd_done_wall_us": int(cmd_done_wall_us),
                "cmd_ret": int(cmd_ret),
            })

    def read_loop() -> None:
        assert source is not None
        t0 = time.time()
        while not stop_evt.is_set():
            raw_preview_frame = None
            raw_preview_meta: Dict[str, Any] = {}
            try:
                if raw_preview_enabled and hasattr(source, "read_with_raw_preview"):
                    ok, frame, ts_us, raw_preview_frame, raw_preview_meta = source.read_with_raw_preview()
                else:
                    ok, frame, ts_us = source.read()
            except Exception as e:
                timeout_total = _inc_stat("timeout_count", 1)
                if timeout_total <= 20:
                    print(f"[mvs_worker][{cam_id}][WARN] read exception: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.001)
                continue

            if not ok or frame is None:
                timeout_total = _inc_stat("timeout_count", 1)
                if timeout_total <= 20:
                    print(f"[mvs_worker][{cam_id}][WARN] no frame", flush=True)
                time.sleep(0.001)
                continue

            meta_from_trigger = _pop_trigger_meta()
            if meta_from_trigger is None:
                orphan_total = _inc_stat("orphan_frame_count", 1)
                if orphan_total <= 20:
                    print(f"[mvs_worker][{cam_id}][WARN] frame received but no pending trigger metadata; skip orphan frame", flush=True)
                continue

            if int(getattr(args, "resize_width", 0)) > 0 and int(getattr(args, "resize_height", 0)) > 0:
                frame = cv2.resize(frame, (int(args.resize_width), int(args.resize_height)), interpolation=cv2.INTER_AREA)

            frame_count = _inc_stat("frame_count", 1)
            _inc_stat("read_ok_total", 1)
            read_ok_window = _inc_stat("read_ok_window", 1)

            tick_id = int(meta_from_trigger.get("tick_id", -1))
            fire_ns = int(meta_from_trigger.get("fire_ns", 0) or 0)
            trigger_call_ns = int(meta_from_trigger.get("trigger_call_ns", 0) or 0)
            period_ns = int(meta_from_trigger.get("period_ns", 0) or 0)
            mvs_frame_num = int(getattr(source, "last_mvs_frame_num", 0) or frame_count)
            mvs_meta = copy.deepcopy(getattr(source, "last_mvs_frame_info", {}) or {})
            exposure_time_us = mvs_meta.get("exposure_time_us", None)
            try:
                exposure_time_us = int(float(exposure_time_us)) if exposure_time_us is not None else None
            except Exception:
                exposure_time_us = None
            exposure_start_offset_us = sync_cfg.get("rgb_exposure_start_offset_us", None)
            try:
                exposure_start_offset_us = int(float(exposure_start_offset_us)) if exposure_start_offset_us is not None else None
            except Exception:
                exposure_start_offset_us = None
            exposure_mid_offset_us = None
            exposure_end_offset_us = None
            if exposure_start_offset_us is not None and exposure_time_us is not None:
                exposure_mid_offset_us = int(exposure_start_offset_us + int(exposure_time_us) // 2)
                exposure_end_offset_us = int(exposure_start_offset_us + int(exposure_time_us))
            mvs_meta.update({
                "program_sync_index": int(tick_id),
                "sync_master": "mvs_soft_trigger",
                "soft_trigger_mode": "external_trigger_coordinator_split_worker_threaded",
                "soft_trigger_period_ms": float(period_ns / 1e6) if period_ns > 0 else None,
                "soft_trigger_fire_time_ns": int(fire_ns),
                "mvs_soft_trigger_est_wall_us": int(meta_from_trigger.get("fire_wall_us_est", 0) or 0),
                "soft_trigger_coord_send_time_ns": int(meta_from_trigger.get("send_time_ns", 0) or 0),
                "soft_trigger_coord_send_wall_us": int(meta_from_trigger.get("send_wall_us", 0) or 0),
                "soft_trigger_worker_trigger_call_ns": int(trigger_call_ns),
                "soft_trigger_worker_trigger_done_ns": int(meta_from_trigger.get("trigger_done_ns", 0) or 0),
                "soft_trigger_worker_delta_us": round((trigger_call_ns - fire_ns) / 1000.0, 3) if fire_ns > 0 and trigger_call_ns > 0 else None,
                "soft_trigger_cmd_wall_us": int(meta_from_trigger.get("cmd_wall_us", 0) or 0),
                "soft_trigger_cmd_done_wall_us": int(meta_from_trigger.get("cmd_done_wall_us", 0) or 0),
                "soft_trigger_cmd_ret": int(meta_from_trigger.get("cmd_ret", 0) or 0),
                "soft_trigger_cmd_ok": bool(meta_from_trigger.get("cmd_ok", False)),
                "receiver_wall_us": int(now_us()),
                "receiver_cam_id": str(cam_id),
                "split_worker_pid": int(os.getpid()),
                "split_worker_threaded": True,
                "split_worker_pending_q": int(_pending_len()),
                "rgb_exposure_as_sync_anchor": False,
            })
            if exposure_start_offset_us is not None:
                mvs_meta["rgb_exposure_start_offset_us"] = int(exposure_start_offset_us)
            if exposure_mid_offset_us is not None:
                mvs_meta["rgb_exposure_mid_offset_us"] = int(exposure_mid_offset_us)
            if exposure_end_offset_us is not None:
                mvs_meta["rgb_exposure_end_offset_us"] = int(exposure_end_offset_us)
            if raw_preview_enabled and raw_preview_frame is not None:
                try:
                    rw = _ensure_raw_preview_writer(raw_preview_frame, raw_preview_meta or {})
                    if rw is not None:
                        rw.write(
                            raw_preview_frame, int(ts_us), int(frame_count), tick_id=int(tick_id),
                            frame_recv_ns=int(mvs_meta.get("receiver_wall_us", 0) or 0) * 1000,
                            trigger_fire_ns=int(fire_ns),
                        )
                except Exception as exc:
                    if int(stats.get("raw_preview_warn_count", 0) or 0) < 10:
                        print(f"[mvs_worker][{cam_id}][WARN] raw preview write failed: {type(exc).__name__}: {exc}", flush=True)
                    stats["raw_preview_warn_count"] = int(stats.get("raw_preview_warn_count", 0) or 0) + 1

            publisher.write(cam_id, frame, int(ts_us), int(frame_count), int(mvs_frame_num), mvs_meta)
            audit.write(cam_id, frame, int(ts_us), int(frame_count), int(mvs_frame_num), mvs_meta)
            if mvs_recorder is not None:
                mvs_recorder.write_frame(cam_id, frame, int(ts_us), int(frame_count), int(mvs_frame_num), mvs_meta)

            if args_cli.print_every > 0 and (read_ok_window % int(args_cli.print_every)) == 0:
                dt = max(1e-6, time.time() - t0)
                snap = _snapshot_stats()
                trigger_ok_window = int(snap.get("trigger_ok_window", 0) or 0)
                print(
                    f"[mvs_worker][{cam_id}] frames={read_ok_window} fps={read_ok_window/dt:.1f} "
                    f"trigger_ok={trigger_ok_window} trigger_fps={trigger_ok_window/dt:.1f} "
                    f"last_tick={tick_id} trig_delta_us={float(snap.get('last_trigger_delta_us', 0.0) or 0.0):.1f} "
                    f"pending_q={_pending_len()} drop={int(snap.get('drop_count', 0) or 0)} "
                    f"timeout={int(snap.get('timeout_count', 0) or 0)} "
                    f"trig_fail={int(snap.get('trigger_fail_count', 0) or 0)} "
                    f"orphan={int(snap.get('orphan_frame_count', 0) or 0)} "
                    f"meta_overflow={int(snap.get('meta_overflow_count', 0) or 0)}",
                    flush=True,
                )
                with stats_lock:
                    stats["read_ok_window"] = 0
                    stats["trigger_ok_window"] = 0
                t0 = time.time()

    trigger_thread: Optional[threading.Thread] = None
    read_thread: Optional[threading.Thread] = None
    try:
        source = _open_source(cam_id, args)
        print(f"[mvs_worker][{cam_id}] opened {source.source_name}; threaded trigger/read worker started", flush=True)

        trigger_thread = threading.Thread(target=trigger_loop, name=f"MvsTrigger-{cam_id}", daemon=True)
        read_thread = threading.Thread(target=read_loop, name=f"MvsRead-{cam_id}", daemon=True)
        trigger_thread.start()
        read_thread.start()

        while not stop_evt.is_set():
            if trigger_thread is not None and not trigger_thread.is_alive():
                print(f"[mvs_worker][{cam_id}][ERROR] trigger thread exited", flush=True)
                stop_evt.set()
                break
            if read_thread is not None and not read_thread.is_alive():
                print(f"[mvs_worker][{cam_id}][ERROR] read thread exited", flush=True)
                stop_evt.set()
                break
            time.sleep(0.2)
    finally:
        stop_evt.set()
        try:
            sock.close()
        except Exception:
            pass
        if trigger_thread is not None:
            trigger_thread.join(timeout=2.0)
        if read_thread is not None:
            read_thread.join(timeout=2.0)
        try:
            if raw_preview_writer is not None:
                raw_preview_writer.close(unlink=False)
        except Exception:
            pass
        try:
            publisher.close()
        except Exception:
            pass
        try:
            audit.close()
        except Exception:
            pass
        try:
            if mvs_recorder is not None:
                mvs_recorder.close()
        except Exception:
            pass
        if source is not None:
            try:
                source.release()
            except Exception:
                pass
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass
        print(f"[mvs_worker][{cam_id}] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
