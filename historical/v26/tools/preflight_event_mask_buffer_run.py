#!/usr/bin/env python3
"""Preflight/health summary for event human-mask buffer evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path


BUFFER_KEYS = (
    "event_human_mask_buffer_enable",
    "event_human_mask_buffer_ms",
    "event_human_mask_buffer_timeout_ms",
    "event_human_mask_buffer_release_policy",
    "event_human_mask_buffer_max_events",
    "event_human_mask_buffer_max_buckets",
    "event_human_mask_buffer_empty_release_fps",
    "event_human_mask_filter_max_fps",
)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _mtime_age_sec(path: Path, now: float) -> float | None:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except Exception:
        return None


def _round_float(v, nd=3):
    try:
        f = float(v)
        if math.isfinite(f):
            return round(f, nd)
    except Exception:
        pass
    return None


def _profile_args(profile):
    if isinstance(profile, dict) and isinstance(profile.get("args"), dict):
        return profile["args"]
    return profile if isinstance(profile, dict) else {}


def _runtime_shared(runtime):
    if not isinstance(runtime, dict):
        return {}
    event_cfg = runtime.get("event_config")
    if isinstance(event_cfg, dict) and isinstance(event_cfg.get("shared_options"), dict):
        return event_cfg["shared_options"]
    if isinstance(runtime.get("shared_options"), dict):
        return runtime["shared_options"]
    return {}


def _expected_settings(path: Path):
    data = _read_json(path) if path else None
    if isinstance(data, dict) and isinstance(data.get("settings"), dict):
        return data["settings"], data
    return {}, data


def _extract_settings(src):
    if not isinstance(src, dict):
        return {}
    return {k: src.get(k) for k in BUFFER_KEYS if k in src}


def _compare_settings(expected, actual):
    out = {}
    ok = True
    for key, exp in expected.items():
        if key not in BUFFER_KEYS:
            continue
        act = actual.get(key)
        same = act == exp
        if isinstance(exp, float) or isinstance(act, float):
            try:
                same = abs(float(act) - float(exp)) <= 1e-6
            except Exception:
                same = False
        if same is False:
            ok = False
        out[key] = {"expected": exp, "actual": act, "ok": bool(same)}
    return ok, out


def _run_probe(cmd, timeout_sec=12.0):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "available": True,
            "rc": proc.returncode,
            "output": proc.stdout[-12000:],
        }
    except FileNotFoundError:
        return {"available": False, "rc": None, "output": ""}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        return {"available": True, "rc": 124, "output": out[-12000:], "timeout": True}
    except Exception as exc:
        return {"available": bool(shutil.which(cmd[0])), "rc": None, "output": str(exc), "error": type(exc).__name__}


def _event_device_probe(enabled: bool):
    if not enabled:
        return {"enabled": False, "ok": None}
    lsusb = _run_probe(["lsusb"], timeout_sec=5.0)
    platform = _run_probe(["metavision_platform_info"], timeout_sec=12.0)
    platform_text = platform.get("output") or ""
    lsusb_text = lsusb.get("output") or ""
    no_systems = "No systems USB connected" in platform_text
    platform_present = platform.get("available") and not no_systems
    usb_hint_present = any(token in lsusb_text.lower() for token in ("prophesee", "metavision"))
    ok = bool(platform_present or usb_hint_present)
    return {
        "enabled": True,
        "ok": ok,
        "metavision_platform_info_available": bool(platform.get("available")),
        "metavision_platform_info_rc": platform.get("rc"),
        "metavision_systems_present": bool(platform_present),
        "metavision_no_systems_message": bool(no_systems),
        "lsusb_available": bool(lsusb.get("available")),
        "lsusb_rc": lsusb.get("rc"),
        "lsusb_event_hint_present": bool(usb_hint_present),
        "platform_info_excerpt": platform_text[-2000:],
        "lsusb_excerpt": lsusb_text[-2000:],
    }


def _status_summary(path: Path, now: float):
    rec = _read_json(path)
    age = _mtime_age_sec(path, now)
    if not isinstance(rec, dict):
        return {"exists": path.exists(), "age_sec": _round_float(age), "parse_ok": False}
    hmb = rec.get("human_mask_buffer") if isinstance(rec.get("human_mask_buffer"), dict) else {}
    hm = rec.get("human_mask") if isinstance(rec.get("human_mask"), dict) else {}
    return {
        "exists": True,
        "age_sec": _round_float(age),
        "parse_ok": True,
        "pid": rec.get("pid"),
        "state": rec.get("state"),
        "active_reason": rec.get("active_reason"),
        "total_slice_count": rec.get("total_slice_count"),
        "buffer": {
            "enabled": hmb.get("enabled"),
            "buffer_ms": hmb.get("buffer_ms"),
            "timeout_ms": hmb.get("timeout_ms"),
            "release_policy": hmb.get("release_policy"),
            "reason": hmb.get("reason"),
            "released_slices": hmb.get("released_slices"),
            "held_slices": hmb.get("held_slices"),
            "legacy_held_slices": hmb.get("legacy_held_slices"),
            "logical_held_slices": hmb.get("logical_held_slices"),
            "empty_skipped_slices": hmb.get("empty_skipped_slices"),
            "forced_releases": hmb.get("forced_releases"),
            "release_counts": hmb.get("release_counts", {}),
        },
        "human_mask": {
            "enabled": hm.get("enabled"),
            "applied": hm.get("applied"),
            "reason": hm.get("reason"),
            "current_sync_index": hm.get("current_sync_index"),
            "buffer_release_reason": hm.get("buffer_release_reason"),
        },
    }


def build_summary(args):
    project_root = Path(args.project_root).resolve()
    sync_root = (project_root / args.sync_root).resolve()
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    now = time.time()

    round_config = Path(args.round_config).resolve() if args.round_config else None
    expected, round_config_data = _expected_settings(round_config)

    profile_path = (project_root / args.profile).resolve() if args.profile else None
    profile = _read_json(profile_path) if profile_path else None
    profile_settings = _extract_settings(_profile_args(profile))

    runtime_path = sync_root / "_runtime_config" / "event_runtime_from_unified.json"
    runtime = _read_json(runtime_path)
    runtime_settings = _extract_settings(_runtime_shared(runtime))

    profile_ok, profile_compare = _compare_settings(expected, profile_settings)
    runtime_ok, runtime_compare = _compare_settings(expected, runtime_settings)

    ready = {}
    human_results = {}
    statuses = {}
    for cam in cams:
        ready_path = sync_root / f"event_ready_{cam}.flag"
        human_path = sync_root / f"human_result_{cam}.jsonl"
        status_path = sync_root / f"event_worker_{cam}_status.json"
        ready[cam] = {
            "exists": ready_path.exists(),
            "age_sec": _round_float(_mtime_age_sec(ready_path, now)),
        }
        human_results[cam] = {
            "exists": human_path.exists(),
            "age_sec": _round_float(_mtime_age_sec(human_path, now)),
            "bytes": human_path.stat().st_size if human_path.exists() else 0,
        }
        statuses[cam] = _status_summary(status_path, now)

    global_status_path = sync_root / "global_yolo_status.json"
    global_status = _read_json(global_status_path)
    yolo = {
        "exists": global_status_path.exists(),
        "age_sec": _round_float(_mtime_age_sec(global_status_path, now)),
    }
    if isinstance(global_status, dict):
        stats = global_status.get("stats") if isinstance(global_status.get("stats"), dict) else {}
        start = stats.get("start_wall_us")
        wall = global_status.get("wall_us")
        frames = stats.get("frames_inferred")
        fps = None
        try:
            elapsed = (float(wall) - float(start)) / 1_000_000.0
            fps = float(frames) / elapsed if elapsed > 0 else None
        except Exception:
            elapsed = None
        yolo.update({
            "state": global_status.get("state"),
            "worker_count": global_status.get("worker_count"),
            "frames_inferred": frames,
            "elapsed_sec": _round_float(elapsed),
            "fps": _round_float(fps),
            "per_cam_fps": _round_float(float(fps) / max(1, len(cams))) if fps is not None else None,
            "tensor_rt_enabled": global_status.get("tensor_rt_enabled", global_status.get("tensorrt_enabled")),
            "legacy_yolo_skipped": global_status.get("legacy_yolo_skipped"),
        })

    event_devices = _event_device_probe(bool(args.probe_event_devices))
    event_devices_ok = True
    if args.require_event_devices:
        event_devices_ok = bool(event_devices.get("ok"))

    max_age = float(args.max_age_sec)
    required_ready = all(item["exists"] and (item["age_sec"] is None or item["age_sec"] <= max_age) for item in ready.values())
    required_human = all(item["exists"] and item["bytes"] > 0 and (item["age_sec"] is None or item["age_sec"] <= max_age) for item in human_results.values())
    required_status = all(item.get("exists") and item.get("parse_ok") and (item.get("age_sec") is None or item.get("age_sec") <= max_age) for item in statuses.values())
    status_params_ok = True
    for item in statuses.values():
        buf = item.get("buffer", {})
        if expected.get("event_human_mask_buffer_enable") is not None and buf.get("enabled") != expected.get("event_human_mask_buffer_enable"):
            status_params_ok = False
        for key, buf_key in (
            ("event_human_mask_buffer_ms", "buffer_ms"),
            ("event_human_mask_buffer_timeout_ms", "timeout_ms"),
            ("event_human_mask_buffer_release_policy", "release_policy"),
        ):
            if key not in expected:
                continue
            exp = expected[key]
            act = buf.get(buf_key)
            same = act == exp
            if isinstance(exp, float) or isinstance(act, float):
                try:
                    same = abs(float(act) - float(exp)) <= 1e-6
                except Exception:
                    same = False
            if not same:
                status_params_ok = False

    ok = bool(
        profile_ok
        and runtime_ok
        and required_ready
        and required_human
        and required_status
        and status_params_ok
        and event_devices_ok
    )
    return {
        "ok": ok,
        "project_root": str(project_root),
        "sync_root": str(sync_root),
        "cams": cams,
        "round_config": str(round_config) if round_config else "",
        "round_config_data": round_config_data,
        "expected_settings": expected,
        "profile": {
            "path": str(profile_path) if profile_path else "",
            "settings": profile_settings,
            "compare": profile_compare,
            "ok": bool(profile_ok),
        },
        "runtime": {
            "path": str(runtime_path),
            "settings": runtime_settings,
            "compare": runtime_compare,
            "ok": bool(runtime_ok),
        },
        "ready": ready,
        "human_results": human_results,
        "event_worker_status": statuses,
        "global_yolo": yolo,
        "event_devices": event_devices,
        "checks": {
            "profile_settings_ok": bool(profile_ok),
            "runtime_settings_ok": bool(runtime_ok),
            "ready_fresh_ok": bool(required_ready),
            "human_results_fresh_ok": bool(required_human),
            "event_worker_status_fresh_ok": bool(required_status),
            "event_worker_buffer_params_ok": bool(status_params_ok),
            "event_devices_ok": bool(event_devices_ok),
        },
    }


def write_markdown(summary, out_md: Path):
    lines = [
        "# Event Mask Buffer Preflight",
        "",
        f"- ok: {summary.get('ok')}",
        f"- project_root: `{summary.get('project_root')}`",
        f"- sync_root: `{summary.get('sync_root')}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in summary.get("checks", {}).items():
        lines.append(f"- {key}: {value}")
    yolo = summary.get("global_yolo", {})
    devices = summary.get("event_devices", {})
    lines += [
        "",
        "## Global YOLO",
        "",
        f"- state: {yolo.get('state')}",
        f"- fps/per_cam: {yolo.get('fps')} / {yolo.get('per_cam_fps')}",
        f"- tensor_rt_enabled: {yolo.get('tensor_rt_enabled')}",
        "",
        "## Event Devices",
        "",
        f"- probe_enabled: {devices.get('enabled')}",
        f"- ok: {devices.get('ok')}",
        f"- metavision_platform_info_available: {devices.get('metavision_platform_info_available')}",
        f"- metavision_systems_present: {devices.get('metavision_systems_present')}",
        f"- metavision_no_systems_message: {devices.get('metavision_no_systems_message')}",
        f"- lsusb_event_hint_present: {devices.get('lsusb_event_hint_present')}",
        "",
        "## Event Workers",
        "",
        "| cam | status_age | ready_age | human_age | human_bytes | state | buffer | reason | forced | held | legacy_held |",
        "|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|",
    ]
    statuses = summary.get("event_worker_status", {})
    ready = summary.get("ready", {})
    human = summary.get("human_results", {})
    for cam in summary.get("cams", []):
        st = statuses.get(cam, {})
        buf = st.get("buffer", {})
        lines.append(
            f"| {cam} | {st.get('age_sec')} | {ready.get(cam, {}).get('age_sec')} | "
            f"{human.get(cam, {}).get('age_sec')} | {human.get(cam, {}).get('bytes')} | "
            f"{st.get('state')} | {buf.get('buffer_ms')}/{buf.get('timeout_ms')} {buf.get('release_policy')} | "
            f"{buf.get('reason')} | {buf.get('forced_releases')} | {buf.get('held_slices')} | "
            f"{buf.get('legacy_held_slices')} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--profile", default="launch_profiles/627_event_overlay_bev_v3_sandbox.json")
    ap.add_argument("--round-config", default="")
    ap.add_argument("--cams", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--max-age-sec", type=float, default=300.0)
    ap.add_argument("--probe-event-devices", action="store_true")
    ap.add_argument("--require-event-devices", action="store_true")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    summary = build_summary(args)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, out_md)
    print(text)
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
