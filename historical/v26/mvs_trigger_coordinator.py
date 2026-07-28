#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish a shared future trigger clock for split MVS camera workers.

The coordinator does NOT open any Hik/MVS camera.  It only sends future
`fire_time_ns` ticks to per-camera Unix datagram sockets.  Each worker owns its
own camera handle and calls TriggerSoftware locally at the announced time.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _as_bool(v, default=False) -> bool:
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


def _merged_mvs_config(path: str) -> Dict:
    cfg = _load_json(path)
    active = str(cfg.get("active_profile", "") or "").strip()
    if active and isinstance(cfg.get("profiles"), dict) and active in cfg["profiles"]:
        import copy
        def deep_update(dst, src):
            for k, v in (src or {}).items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_update(dst[k], v)
                else:
                    dst[k] = copy.deepcopy(v)
            return dst
        cfg = deep_update(copy.deepcopy(cfg), cfg["profiles"].get(active) or {})
        cfg["active_profile"] = active
    return cfg


def _parse_cams(cams: str, cfg: Dict) -> List[str]:
    if cams:
        return [x.strip() for x in cams.split(",") if x.strip()]
    out: List[str] = []
    for r in cfg.get("routes", []) or []:
        if not isinstance(r, dict) or not bool(r.get("enabled", True)):
            continue
        cid = str(r.get("id", r.get("name", "")) or "").strip()
        if cid:
            out.append(cid)
    return out


def _sock_dir(sync_cfg: Dict, override: str = "") -> Path:
    root = str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
    d = str(override or sync_cfg.get("mvs_trigger_ipc_dir", "{root}/trigger_ipc") or "{root}/trigger_ipc")
    d = d.format(root=root)
    return Path(d).expanduser().resolve()


def _socket_path(sock_dir: Path, cam: str, template: str) -> Path:
    return sock_dir / template.format(camera=str(cam), cam=str(cam), id=str(cam))


def _sleep_until_ns(target_ns: int) -> None:
    while True:
        now = time.monotonic_ns()
        remain = int(target_ns) - now
        if remain <= 0:
            return
        if remain > 2_000_000:
            time.sleep(min(0.002, remain / 1e9))
        else:
            time.sleep(remain / 1e9)


def _write_sync_start_file(path: str, cams: List[str], tick_id: int, fps: float, period_ns: int) -> str:
    raw = str(path or '').strip()
    if not raw:
        return ''
    p = Path(raw).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sync_start": True,
        "tick_id_start": int(tick_id),
        "cams": list(cams),
        "fps": float(fps),
        "period_ns": int(period_ns),
        "wall_time_us": int(time.time() * 1_000_000),
        "monotonic_ns": int(time.monotonic_ns()),
        "pid": int(os.getpid()),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(p)
    print(f"[trigger_coord] sync_start_file written -> {p} tick_id_start={int(tick_id)}", flush=True)
    return str(p)




def _pctl(values, q: float) -> float:
    vals = sorted(float(x) for x in list(values) if x is not None)
    if not vals:
        return 0.0
    idx = int(round(float(q) * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _p95(values) -> float:
    return _pctl(values, 0.95)


def _write_trigger_status(path: str, payload: Dict) -> None:
    raw = str(path or '').strip()
    if not raw:
        return
    try:
        p = Path(raw).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
        tmp.replace(p)
    except Exception:
        pass


def _format_sync_path(sync_cfg: Dict, key: str, default_template: str) -> str:
    root = str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
    raw = str(sync_cfg.get(key, default_template) or default_template)
    raw = raw.format(root=root)
    return str(Path(raw).expanduser().resolve())


def _open_timeline_writer(path: str):
    raw = str(path or "").strip()
    if not raw:
        return None
    try:
        p = Path(raw).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, "w", encoding="utf-8", buffering=1)
    except Exception as exc:
        print(f"[trigger_coord][WARN] open timeline failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _write_timeline_latest(path: str, payload: Dict) -> None:
    raw = str(path or "").strip()
    if not raw:
        return
    try:
        p = Path(raw).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="MVS split-worker trigger clock coordinator")
    ap.add_argument("--config", required=True, help="MVS runtime JSON")
    ap.add_argument("--cams", default="", help="Comma separated cams. Default: enabled routes in config")
    ap.add_argument("--fps", type=float, default=0.0, help="Trigger FPS. Default: sync.mvs_soft_trigger_fps or 1000/period_ms")
    ap.add_argument("--lead-ms", type=float, default=-1.0, help="Send each tick this many ms before fire_time. Default config/env or 5ms")
    ap.add_argument("--start-delay-ms", type=float, default=1000.0, help="Delay before first fire_time after workers are ready")
    ap.add_argument("--sync-start-file", default="", help="Write this flag after start-delay and before tick_id=0 so event cameras reset their sync_index")
    ap.add_argument("--sync-start-settle-ms", type=float, default=1000.0, help="Wait this many ms after writing sync-start-file before the first tick")
    ap.add_argument("--status-file", default="", help="Write trigger telemetry JSON. Default: <sync.root>/trigger_status.json")
    ap.add_argument("--timeline-file", default="", help="Write authoritative MVS soft-trigger timeline JSONL. Default: <sync.root>/global_trigger_timeline.jsonl")
    ap.add_argument("--timeline-latest-file", default="", help="Write latest MVS soft-trigger timeline JSON. Default: <sync.root>/global_trigger_timeline_latest.json")
    ap.add_argument("--socket-dir", default="", help="Override trigger IPC socket directory")
    ap.add_argument("--socket-template", default="trigger_{camera}.sock")
    ap.add_argument("--wait-workers", action="store_true", default=True)
    ap.add_argument("--no-wait-workers", dest="wait_workers", action="store_false")
    ap.add_argument("--ready-timeout", type=float, default=20.0)
    ap.add_argument("--late-resync-periods", type=float, default=3.0)
    ap.add_argument("--print-every", type=int, default=40)
    args = ap.parse_args()

    cfg = _merged_mvs_config(args.config)
    sync_cfg = cfg.get("sync", {}) if isinstance(cfg.get("sync", {}), dict) else {}
    cams = _parse_cams(args.cams, cfg)
    if not cams:
        raise SystemExit("[trigger_coord][ERROR] no cameras")

    period_ms = float(sync_cfg.get("mvs_soft_trigger_period_ms", 25.0) or 25.0)
    fps_cfg = float(sync_cfg.get("mvs_soft_trigger_fps", 0.0) or 0.0)
    fps = float(args.fps or fps_cfg or (1000.0 / max(0.001, period_ms)))
    fps = max(1.0, min(500.0, fps))
    period_ns = int(round(1_000_000_000.0 / fps))
    lead_ms = float(args.lead_ms if args.lead_ms >= 0 else sync_cfg.get("mvs_trigger_coord_lead_ms", 5.0) or 5.0)
    lead_ns = int(max(0.0, lead_ms) * 1_000_000.0)
    sock_dir = _sock_dir(sync_cfg, args.socket_dir)
    sock_dir.mkdir(parents=True, exist_ok=True)
    paths = {c: _socket_path(sock_dir, c, args.socket_template) for c in cams}
    root = str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
    status_file = str(args.status_file or str(Path(root).expanduser().resolve() / "trigger_status.json"))
    timeline_file = str(args.timeline_file or _format_sync_path(sync_cfg, "global_trigger_timeline_jsonl", "{root}/global_trigger_timeline.jsonl"))
    timeline_latest_file = str(args.timeline_latest_file or _format_sync_path(sync_cfg, "global_trigger_timeline_latest_json", "{root}/global_trigger_timeline_latest.json"))
    timeline_fp = _open_timeline_writer(timeline_file)
    jitter_ms_window = deque(maxlen=512)
    last_status_wall = 0.0

    print(f"[trigger_coord] cams={','.join(cams)} fps={fps:.3f} period_ms={period_ns/1e6:.3f} lead_ms={lead_ms:.3f}", flush=True)
    print(f"[trigger_coord] socket_dir={sock_dir}", flush=True)
    if timeline_fp is not None:
        print(f"[trigger_coord] MVS soft-trigger timeline -> {timeline_file}", flush=True)
    _write_trigger_status(status_file, {
        "state": "waiting_workers",
        "fps": float(fps),
        "trigger_fps": 0.0,
        "period_ms": float(period_ns / 1e6),
        "lead_ms": float(lead_ms),
        "ok_cams": 0,
        "total_cams": len(cams),
        "jitter_p95_ms": 0.0,
        "wall_time_us": int(time.time() * 1_000_000),
    })

    if args.wait_workers:
        deadline = time.time() + float(args.ready_timeout)
        missing = list(cams)
        while time.time() < deadline:
            missing = [c for c, p in paths.items() if not p.exists()]
            if not missing:
                break
            time.sleep(0.05)
        if missing:
            print(f"[trigger_coord][WARN] worker socket wait timeout; missing={','.join(missing)}", flush=True)
        else:
            print("[trigger_coord] all worker sockets ready", flush=True)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    running = True

    def _stop(signum=None, frame=None):
        nonlocal running
        running = False
    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    tick_id = int(sync_cfg.get("mvs_soft_trigger_start_index", 0) or 0)
    if str(args.sync_start_file or '').strip():
        delay_ms = max(0.0, float(args.start_delay_ms or 0.0))
        if delay_ms > 0.0:
            print(f"[trigger_coord] waiting {delay_ms:.1f} ms before sync_start_file", flush=True)
            _write_trigger_status(status_file, {
                "state": "waiting_sync_start",
                "fps": float(fps),
                "trigger_fps": 0.0,
                "period_ms": float(period_ns / 1e6),
                "lead_ms": float(lead_ms),
                "ok_cams": 0,
                "total_cams": len(cams),
                "jitter_p95_ms": 0.0,
                "wall_time_us": int(time.time() * 1_000_000),
            })
            time.sleep(delay_ms / 1000.0)
        _write_sync_start_file(args.sync_start_file, cams, tick_id, fps, period_ns)
        settle_ms = max(0.0, float(args.sync_start_settle_ms or 0.0))
        if settle_ms > 0.0:
            print(f"[trigger_coord] waiting {settle_ms:.1f} ms after sync_start_file before tick_id={tick_id}", flush=True)
            time.sleep(settle_ms / 1000.0)
        # Make sure workers still receive each tick before fire_time even after the settle sleep.
        next_fire_ns = time.monotonic_ns() + max(int(lead_ns) + 2_000_000, 10_000_000)
    else:
        next_fire_ns = time.monotonic_ns() + int(float(args.start_delay_ms) * 1_000_000.0)
    late_warn = 0
    sent = 0
    fail_count = 0
    per_cam_sent = {c: 0 for c in cams}
    per_cam_fail = {c: 0 for c in cams}
    t0 = time.time()
    while running:
        send_ns = int(next_fire_ns) - lead_ns
        _sleep_until_ns(send_ns)
        now_ns = time.monotonic_ns()
        jitter_ms_window.append((now_ns - send_ns) / 1e6)
        if now_ns > next_fire_ns + int(args.late_resync_periods * period_ns):
            if late_warn < 20:
                print(f"[trigger_coord][WARN] late by {(now_ns-next_fire_ns)/1e6:.2f} ms; resync next_fire", flush=True)
                late_warn += 1
            next_fire_ns = now_ns + period_ns
            tick_id += 1
            continue
        send_wall_us = int(time.time() * 1_000_000)
        fire_wall_us_est = int(send_wall_us + (int(next_fire_ns) - int(now_ns)) / 1000.0)
        msg = {
            "type": "mvs_trigger_tick",
            "tick_id": int(tick_id),
            "fire_time_ns": int(next_fire_ns),
            "fire_wall_us_est": int(fire_wall_us_est),
            "period_ns": int(period_ns),
            "send_time_ns": int(now_ns),
            "send_wall_us": int(send_wall_us),
            "fps": float(fps),
        }
        timeline_rec = {
            "msg_type": "mvs_soft_trigger_tick",
            "sync_master": "mvs_soft_trigger",
            "program_sync_index": int(tick_id),
            "trigger_seq": int(tick_id),
            "mvs_soft_trigger_fire_time_ns": int(next_fire_ns),
            "mvs_soft_trigger_est_wall_us": int(fire_wall_us_est),
            "coord_send_time_ns": int(now_ns),
            "coord_send_wall_us": int(send_wall_us),
            "period_ns": int(period_ns),
            "period_ms": float(period_ns / 1e6),
            "fps": float(fps),
            "lead_ms": float(lead_ms),
            "cams": list(cams),
            "pid": int(os.getpid()),
        }
        data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        ok = 0
        for cam, path in paths.items():
            try:
                sock.sendto(data, str(path))
                ok += 1
                per_cam_sent[cam] = int(per_cam_sent.get(cam, 0)) + 1
            except Exception as e:
                fail_count += 1
                per_cam_fail[cam] = int(per_cam_fail.get(cam, 0)) + 1
                if fail_count <= 20:
                    print(f"[trigger_coord][{cam}][WARN] send failed: {type(e).__name__}: {e}", flush=True)
        timeline_rec["ok_cams"] = int(ok)
        timeline_rec["total_cams"] = int(len(cams))
        if timeline_fp is not None:
            try:
                timeline_fp.write(json.dumps(timeline_rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception:
                pass
        _write_timeline_latest(timeline_latest_file, timeline_rec)
        sent += ok
        if args.print_every > 0 and (tick_id % int(args.print_every)) == 0:
            now_wall = time.time()
            dt = max(1e-6, now_wall - t0)
            sent_rate = sent / dt
            trig_fps = sent_rate / max(1, ok)
            p50 = _pctl(jitter_ms_window, 0.50)
            p95 = _p95(jitter_ms_window)
            jmax = max([float(x) for x in jitter_ms_window], default=0.0)
            print(f"[trigger_coord] tick={tick_id} ok_cams={ok}/{len(cams)} sent_rate={sent_rate:.1f}/s trigger_fps={trig_fps:.2f} jitter_p95_ms={p95:.3f}", flush=True)
            _write_trigger_status(status_file, {
                "state": "running" if ok == len(cams) else "degraded",
                "tick_id": int(tick_id),
                "fps": float(fps),
                "trigger_fps": float(trig_fps),
                "period_ms": float(period_ns / 1e6),
                "lead_ms": float(lead_ms),
                "ok_cams": int(ok),
                "total_cams": len(cams),
                "sent_rate": float(sent_rate),
                "jitter_p50_ms": float(p50),
                "jitter_p95_ms": float(p95),
                "jitter_max_ms": float(jmax),
                "last_send_jitter_ms": float(jitter_ms_window[-1]) if jitter_ms_window else 0.0,
                "per_cam_sent": dict(per_cam_sent),
                "per_cam_fail": dict(per_cam_fail),
                "wall_time_us": int(now_wall * 1_000_000),
                "monotonic_ns": int(now_ns),
            })
            sent = 0
            per_cam_sent = {c: 0 for c in cams}
            per_cam_fail = {c: 0 for c in cams}
            t0 = now_wall
        tick_id += 1
        next_fire_ns += period_ns
    try:
        sock.close()
    except Exception:
        pass
    try:
        if timeline_fp is not None:
            timeline_fp.close()
    except Exception:
        pass
    print("[trigger_coord] stopped", flush=True)
    _write_trigger_status(status_file, {"state": "stopped", "wall_time_us": int(time.time() * 1_000_000)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
