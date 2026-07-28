#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_CONTROL_STATES = {"live_ready", "ok", "degraded"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_pid(path: Path) -> int:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ProcessLookupError, PermissionError, TypeError, ValueError):
        return 0


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _tail(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _status_base(control: Dict[str, Any], label: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "v25_panel_launch_status",
        "ts_wall_us": int(time.time() * 1_000_000),
        "state": "starting",
        "error": "",
        "label": label,
        "mode": str(control.get("mode") or ""),
        "profile": str(control.get("profile") or ""),
        "dataset_dir": str(control.get("dataset_dir") or ""),
        "replay_timing": str(control.get("replay_timing") or ""),
        "control_state": str(control.get("state") or ""),
        "control_ts_wall_us": control.get("ts_wall_us"),
        "launch_extra_argv": control.get("launch_extra_argv", []),
        "launcher_pid": 0,
        "run_dir": "",
        "start_returncode": None,
        "ready_returncode": None,
    }


def _load_extra_argv(control: Dict[str, Any]) -> List[str]:
    value = control.get("launch_extra_argv")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    legacy = str(control.get("launch_extra_args") or "").strip()
    return shlex.split(legacy) if legacy else []


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Start the managed full system from V25 panel state.")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--control", default="sync_ipc/v25_replay_status.json")
    ap.add_argument("--status", default="sync_ipc/v25_panel_launch_status.json")
    ap.add_argument("--label", default="desktop_latest")
    ap.add_argument("--ready-wait", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    control_path = Path(args.control).expanduser()
    if not control_path.is_absolute():
        control_path = project_root / control_path
    status_path = Path(args.status).expanduser()
    if not status_path.is_absolute():
        status_path = project_root / status_path

    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = _status_base({}, args.label)
        payload.update(state="error", error=f"control_read_failed: {type(exc).__name__}: {exc}")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    payload = _status_base(control, args.label)
    control_state = str(control.get("state") or "")
    if control_state not in ALLOWED_CONTROL_STATES:
        payload.update(state="blocked", error=f"control_state_not_startable: {control_state or 'missing'}")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3
    if control_state == "degraded" and not bool(control.get("allow_missing_ext_trigger")):
        payload.update(state="blocked", error="degraded_control_requires_explicit_allow_missing_ext_trigger")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    profile = str(control.get("profile") or "").strip()
    profile_path = Path(profile).expanduser()
    if not profile_path.is_absolute():
        profile_path = project_root / profile_path
    if not profile or not profile_path.is_file():
        payload.update(state="blocked", error=f"profile_not_found: {profile_path}")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    mode = str(control.get("mode") or "")
    if mode not in {"live", "event_replay", "mvs_replay", "full_replay"}:
        payload.update(state="blocked", error=f"invalid_mode: {mode or 'missing'}")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3
    dataset_dir = str(control.get("dataset_dir") or "").strip()
    if mode != "live" and (not dataset_dir or not Path(dataset_dir).is_dir()):
        payload.update(state="blocked", error=f"dataset_dir_not_found: {dataset_dir or 'missing'}")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    extra_argv = _load_extra_argv(control)
    payload["launch_extra_argv"] = extra_argv
    payload["launch_command"] = shlex.join([
        sys.executable,
        "launch_fusion_system.py",
        "--launch-profile",
        profile,
        *extra_argv,
    ])
    if args.dry_run:
        payload.update(state="dry_run_ok", ts_wall_us=int(time.time() * 1_000_000))
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _write_json(status_path, payload)
    env = os.environ.copy()
    env.update({
        "PROJECT": str(project_root),
        "PY": sys.executable,
        "PROFILE": profile,
        "LABEL": args.label,
        "LAUNCH_EXTRA_ARGS_JSON": json.dumps(extra_argv, ensure_ascii=False),
        "V25_DATASET_DIR": dataset_dir,
        "MVS_SOURCE_MODE_REQUESTED": "file_replay" if mode in {"mvs_replay", "full_replay"} else "live_camera",
    })

    pid_file = project_root / "sync_ipc" / f"{args.label}_launcher.pid"
    latest_file = project_root / "sync_ipc" / f"latest_{args.label}_run_dir.txt"
    was_running = _read_pid(pid_file) > 0
    start = subprocess.run(
        ["bash", "remote_ops/start_latest_system.sh"],
        cwd=project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(start.stdout, end="")
    payload.update({
        "ts_wall_us": int(time.time() * 1_000_000),
        "start_returncode": start.returncode,
        "start_output_tail": _tail(start.stdout),
        "launcher_pid": _read_pid(pid_file),
        "run_dir": _read_text(latest_file),
    })
    if start.returncode != 0:
        payload.update(state="error", error="managed_start_failed")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return start.returncode or 4

    if args.ready_wait <= 0:
        payload.update(state="already_running" if was_running else "started")
        _write_json(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    ready = subprocess.run(
        ["bash", "remote_ops/wait_latest_system_ready.sh", str(args.ready_wait)],
        cwd=project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(ready.stdout, end="")
    payload.update({
        "ts_wall_us": int(time.time() * 1_000_000),
        "ready_returncode": ready.returncode,
        "ready_output_tail": _tail(ready.stdout),
        "launcher_pid": _read_pid(pid_file),
        "run_dir": _read_text(latest_file),
    })
    if ready.returncode == 0:
        payload.update(state="already_running_ready" if was_running else "ready", error="")
        rc = 0
    else:
        payload.update(state="running_not_ready", error="readiness_check_failed")
        rc = 5
    _write_json(status_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
