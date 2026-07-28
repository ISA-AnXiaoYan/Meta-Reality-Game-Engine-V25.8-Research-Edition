#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-click launcher for the MVS + event + hit judge + OpenGL preview stack.

Start from the project root:
    python3 launch_fusion_system.py --cams A,B,C,D,E

It starts, in order:
  1) event cameras via multi_camera_launcher.py
  2) MVS/YOLO/human-region publisher
  3) impact hit judge server
  4) OpenGL fusion renderer

Ctrl+C terminates all child process groups.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class ManagedProc:
    """Managed child process with throttled console output.

    Key design:
      - Always consumes child stdout so the child will not block on a full pipe.
      - Writes full child logs to sync_ipc/launcher_logs/*.log.
      - Prints only startup lines and important lines to PyCharm/terminal by default.
    """

    IMPORTANT_KEYWORDS = (
        "error",
        "warn",
        "warning",
        "traceback",
        "exception",
        "failed",
        "fail",
        "fatal",
        "exited",
        "ready",
        "listen",
        "listening",
        "started",
        "client connected",
        "vpoint",
        "vpoint_write",
        "vpoint_fast_write",
        "vpoint_drop",
        "vpoint_dup_drop",
        "vpoint_fast_error",
        "hit_judge",
        "websocket",
        "ws://",
        "udp",
    )

    def __init__(
        self,
        name: str,
        cmd: List[str],
        cwd: Path,
        critical: bool = True,
        log_dir: Optional[Path] = None,
        console_mode: str = "startup",
        startup_lines: int = 60,
    ):
        self.name = name
        self.cmd = [str(x) for x in cmd]
        self.cwd = Path(cwd)
        self.critical = bool(critical)
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None

        self.console_mode = str(console_mode or "startup").lower().strip()
        if self.console_mode not in {"all", "startup", "errors", "quiet"}:
            self.console_mode = "startup"
        self.startup_lines = max(0, int(startup_lines))

        if log_dir is None:
            log_dir = self.cwd / "sync_ipc" / "launcher_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{self.name.lower()}.log"

    def start(self) -> None:
        print(f"[launcher] start {self.name}: {shlex.join(self.cmd)}", flush=True)
        print(f"[launcher] {self.name} full log -> {self.log_path}", flush=True)
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name == 'posix' else None,
        )
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _should_print_to_console(self, line_no: int, text: str) -> bool:
        if self.console_mode == "all":
            return True
        if self.console_mode == "quiet":
            return False

        low = text.lower()
        important = any(k in low for k in self.IMPORTANT_KEYWORDS)

        if self.console_mode == "errors":
            return important

        # startup mode: show the beginning of each child process, then only key lines.
        if line_no < self.startup_lines:
            return True
        return important

    def _pump(self) -> None:
        if not self.proc or not self.proc.stdout:
            return

        line_no = 0
        try:
            # Overwrite each child log on each new run, avoiding huge accumulated logs.
            with open(self.log_path, "w", encoding="utf-8", buffering=1) as lf:
                lf.write("=" * 80 + "\n")
                lf.write(f"[launcher] child={self.name}\n")
                lf.write(f"[launcher] cwd={self.cwd}\n")
                lf.write(f"[launcher] cmd={shlex.join(self.cmd)}\n")
                lf.write("=" * 80 + "\n")

                for line in self.proc.stdout:
                    text = line.rstrip("\r\n")

                    # Important: always drain stdout, even when console is quiet.
                    lf.write(f"[{self.name}] {text}\n")

                    if self._should_print_to_console(line_no, text):
                        print(f"[{self.name}] {text}", flush=True)

                    line_no += 1
        except Exception as exc:
            print(f"[launcher][WARN] log pump failed for {self.name}: {exc}", flush=True)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def terminate(self, timeout: float = 5.0) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        print(f"[launcher] terminate {self.name}", flush=True)
        try:
            if os.name == 'posix':
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.terminate()
        except Exception as exc:
            print(f"[launcher][WARN] terminate {self.name}: {exc}", flush=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return
            time.sleep(0.1)
        print(f"[launcher] kill {self.name}", flush=True)
        try:
            if os.name == 'posix':
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            else:
                self.proc.kill()
        except Exception:
            pass



def load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def _is_unified_fusion_config(cfg: Dict) -> bool:
    return isinstance(cfg, dict) and isinstance(cfg.get("event_config"), dict) and isinstance(cfg.get("mvs_config"), dict)


def _inject_event_sync_start_options(event_cfg: Dict, project_root: Path) -> Dict:
    """Force event cameras to wait for the formal sync_start flag.

    This prevents pre-start MVS Line/Trigger-In edges from consuming EVENT
    sync_index before the coordinator starts MVS tick_id=0.
    """
    if not isinstance(event_cfg, dict):
        return event_cfg
    shared = event_cfg.setdefault('shared_options', {})
    sync_start_path = str((project_root / 'sync_ipc' / 'sync_start.flag').resolve())
    shared['ext_trigger_sync_start_file'] = sync_start_path
    shared['ext_trigger_require_sync_start'] = True
    shared['ext_trigger_sync_start_reset_log'] = True
    return event_cfg


def _normalize_event_obs_srt_ports(event_cfg: Dict, base_port: int = 6101) -> List[str]:
    if not isinstance(event_cfg, dict):
        return []
    cameras_cfg = event_cfg.get("cameras")
    if not isinstance(cameras_cfg, list):
        return []

    used: set[int] = set()
    next_port = int(base_port)
    changes: List[str] = []
    for idx, cam_cfg in enumerate(cameras_cfg):
        if not isinstance(cam_cfg, dict):
            continue
        if not bool(cam_cfg.get("enable_obs_stream", False)):
            continue
        alias = str(cam_cfg.get("alias") or cam_cfg.get("camera") or cam_cfg.get("name") or idx)
        try:
            port = int(cam_cfg.get("obs_srt_port", 0) or 0)
        except Exception:
            port = 0
        if port <= 0:
            continue
        if port in used:
            while next_port in used:
                next_port += 1
            old_port = port
            port = next_port
            cam_cfg["obs_srt_port"] = port
            changes.append(f"{alias}:{old_port}->{port}")
        used.add(port)
        next_port = max(next_port, port + 1)
    return changes


def _write_launcher_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_launcher_path(project_root: Path, raw: str) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _v25_replay_mode_from_args(args: argparse.Namespace) -> str:
    event_mode = str(getattr(args, "event_source_mode", "live_hal") or "live_hal").strip().lower()
    mvs_mode = str(getattr(args, "mvs_source_mode", "live_camera") or "live_camera").strip().lower()
    if event_mode == "file_replay" and mvs_mode == "file_replay":
        return "full_replay"
    if event_mode == "file_replay":
        return "event_replay"
    if mvs_mode == "file_replay":
        return "mvs_replay"
    return "live"


def _v25_usable_key_for_mode(mode: str) -> str:
    if mode == "event_replay":
        return "usable_for_event_file_replay"
    if mode == "mvs_replay":
        return "usable_for_mvs_file_replay"
    if mode == "full_replay":
        return "usable_for_full_replay"
    return ""


def _parse_camera_mapping_count(raw: str) -> int:
    out = 0
    for part in str(raw or "").split(","):
        if "=" in part and part.split("=", 1)[0].strip():
            out += 1
    return out


def _apply_v25_dataset_source_mode(args: argparse.Namespace, project_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_raw = str(getattr(args, "dataset_dir", "") or "").strip()
    if not dataset_raw:
        return {}, {}
    dataset_dir = _resolve_launcher_path(project_root, dataset_raw)
    mode = _v25_replay_mode_from_args(args)
    if mode == "live":
        return {}, {}
    try:
        from remote_ops.check_v25_dataset_manifest import (
            build_report,
            build_resolved_manifest,
            event_raw_mapping,
        )
    except Exception as exc:
        raise SystemExit(f"[launcher][ERROR] cannot import V25 dataset checker: {type(exc).__name__}: {exc}")

    report = build_report(dataset_dir, mode)
    validation = report.get("validation", {}) if isinstance(report.get("validation"), dict) else {}
    usable_key = _v25_usable_key_for_mode(mode)
    if usable_key and not bool(validation.get(usable_key)):
        raise SystemExit(
            "[launcher][ERROR] V25 dataset is not usable for "
            f"{mode}: key={usable_key} validation={validation} "
            f"missing={report.get('missing', [])} warnings={report.get('warnings', [])}"
        )

    if str(getattr(args, "event_source_mode", "") or "").strip().lower() == "file_replay":
        if not str(getattr(args, "event_file_fifo_broker_raw_files", "") or "").strip():
            mapping = event_raw_mapping(report)
            if mapping:
                setattr(args, "event_file_fifo_broker_raw_files", mapping)
    if str(getattr(args, "mvs_source_mode", "") or "").strip().lower() == "file_replay":
        if not str(getattr(args, "mvs_file_frame_broker_dataset_dir", "") or "").strip():
            source_roots = report.get("source_roots", {}) if isinstance(report.get("source_roots"), dict) else {}
            setattr(args, "mvs_file_frame_broker_dataset_dir", str(source_roots.get("mvs_raw") or dataset_dir))

    replay_timing = (
        str(getattr(args, "event_file_fifo_broker_replay_timing", "") or "")
        or str(getattr(args, "mvs_file_frame_broker_replay_timing", "") or "")
        or "realtime"
    )
    resolved = build_resolved_manifest(
        report,
        replay_timing=replay_timing,
        allow_missing_ext_trigger=bool(getattr(args, "event_file_fifo_broker_allow_missing_ext_trigger", False)),
        event_loop=bool(getattr(args, "event_file_fifo_broker_loop", False)),
        mvs_loop=bool(getattr(args, "mvs_file_frame_broker_loop", False)),
    )
    runtime_dir = project_root / "sync_ipc" / "_runtime_config"
    _write_launcher_json(runtime_dir / "v25_resolved_manifest.json", resolved)
    _write_launcher_json(project_root / "sync_ipc" / "v25_resolved_manifest_latest.json", resolved)
    return report, resolved


def _write_source_mode_effective(
    args: argparse.Namespace,
    project_root: Path,
    applied_profile: Path,
    cams: List[str],
    report: Dict[str, Any],
    resolved: Dict[str, Any],
) -> Dict[str, Any]:
    event_mode = str(getattr(args, "event_source_mode", "live_hal") or "live_hal").strip().lower()
    mvs_mode = str(getattr(args, "mvs_source_mode", "live_camera") or "live_camera").strip().lower()
    dataset_dir = str(getattr(args, "dataset_dir", "") or "").strip()
    event_raw_mapping = str(getattr(args, "event_file_fifo_broker_raw_files", "") or "").strip()
    mvs_dataset_dir = str(getattr(args, "mvs_file_frame_broker_dataset_dir", "") or "").strip()
    validation = report.get("validation", {}) if isinstance(report.get("validation"), dict) else {}
    capabilities = resolved.get("capabilities", {}) if isinstance(resolved.get("capabilities"), dict) else {}
    payload = {
        "schema_version": 1,
        "kind": "v25_source_mode_effective",
        "ts_wall_us": int(time.time() * 1_000_000),
        "profile": str(applied_profile) if applied_profile else "",
        "cams": list(cams),
        "dataset_dir": dataset_dir,
        "dataset_id": str(resolved.get("dataset_id") or report.get("dataset_id") or ""),
        "session_id": str(resolved.get("session_id") or report.get("session_id") or ""),
        "event_source_mode_effective": event_mode,
        "mvs_source_mode_effective": mvs_mode,
        "event_file_fifo_broker_enabled": event_mode == "file_replay",
        "event_live_hal_enabled": event_mode == "live_hal",
        "mvs_file_frame_broker_enabled": mvs_mode == "file_replay",
        "mvs_live_camera_enabled": mvs_mode == "live_camera",
        "mutual_exclusion": {
            "event_file_replay_disables_live_hal": event_mode == "file_replay",
            "event_live_hal_disables_file_replay": event_mode == "live_hal",
            "mvs_file_replay_disables_live_camera": mvs_mode == "file_replay",
            "mvs_live_camera_disables_file_replay": mvs_mode == "live_camera",
        },
        "event_file_fifo_broker_raw_camera_count": _parse_camera_mapping_count(event_raw_mapping),
        "event_file_fifo_broker_raw_files": event_raw_mapping,
        "mvs_file_frame_broker_dataset_dir": mvs_dataset_dir,
        "event_file_fifo_broker_replay_timing": str(getattr(args, "event_file_fifo_broker_replay_timing", "") or ""),
        "mvs_file_frame_broker_replay_timing": str(getattr(args, "mvs_file_frame_broker_replay_timing", "") or ""),
        "event_file_fifo_broker_loop": bool(getattr(args, "event_file_fifo_broker_loop", False)),
        "mvs_file_frame_broker_loop": bool(getattr(args, "mvs_file_frame_broker_loop", False)),
        "validation": validation,
        "capabilities": capabilities,
        "warnings": list(report.get("warnings", [])) if isinstance(report.get("warnings"), list) else [],
        "missing": list(report.get("missing", [])) if isinstance(report.get("missing"), list) else [],
    }
    runtime_dir = project_root / "sync_ipc" / "_runtime_config"
    _write_launcher_json(runtime_dir / "source_mode_effective.json", payload)
    _write_launcher_json(project_root / "sync_ipc" / "source_mode_effective.json", payload)
    return payload


def _write_runtime_split_configs(unified_path: Path, unified_cfg: Dict, project_root: Path) -> tuple[Path, Path, Dict]:
    """Split the single user-facing fusion JSON into event and MVS JSON files.

    The event camera launcher and the MVS/YOLO/hit/OpenGL processes historically
    consume different JSON schemas.  To keep the operator workflow simple we now
    keep exactly one editable config file and generate the two runtime files under
    sync_ipc/_runtime_config/ before starting child processes.
    """
    out_dir = project_root / 'sync_ipc' / '_runtime_config'
    out_dir.mkdir(parents=True, exist_ok=True)
    event_cfg = dict(unified_cfg.get('event_config') or {})
    mvs_cfg = dict(unified_cfg.get('mvs_config') or {})

    event_path = out_dir / 'event_runtime_from_unified.json'
    mvs_path = out_dir / 'mvs_runtime_from_unified.json'

    # Make sync.root absolute. Some consumers, notably fusion_renderer_gl.py,
    # chdir to the runtime-config directory before resolving sync.root.
    sync_cfg = mvs_cfg.setdefault('sync', {})
    root_raw = str(sync_cfg.get('root', './sync_ipc') or './sync_ipc')
    root_path = Path(root_raw).expanduser()
    if not root_path.is_absolute():
        root_path = project_root / root_path
    sync_cfg['root'] = str(root_path.resolve())

    # ---- YSXQ one-key record path fix begin ----
    # Force all children to use the same stable absolute command/status files.
    # This avoids cwd-dependent split-brain:
    #   GL cwd = sync_ipc/_runtime_config
    #   EVENT/MVS cwd = project root
    record_cmd_path = str((project_root / 'sync_ipc' / 'record_control.json').resolve())
    record_status_path = str((project_root / 'sync_ipc' / 'record_status.json').resolve())

    shared = event_cfg.setdefault('shared_options', {})
    shared['enable_remote_raw_record_control'] = True
    shared['raw_record_command_file'] = record_cmd_path
    shared['raw_record_dir'] = str((project_root / 'raw_records').resolve())
    shared['raw_record_prefix'] = str(shared.get('raw_record_prefix') or 'eventraw')
    shared['raw_record_command_poll_ms'] = int(shared.get('raw_record_command_poll_ms') or 5)
    shared['event_native_hal_broker_raw_record_command_file'] = record_cmd_path
    shared['event_native_hal_broker_raw_record_dir'] = str((project_root / 'raw_records').resolve())
    shared['event_native_hal_broker_raw_record_prefix'] = str(shared.get('raw_record_prefix') or 'eventraw')
    shared['event_native_hal_broker_raw_record_poll_ms'] = int(shared.get('raw_record_command_poll_ms') or 5)
    shared['event_slice_record_enable'] = True
    shared['event_slice_record_root'] = str((project_root / 'recordings').resolve())

    one_key_record = mvs_cfg.setdefault('one_key_record', {})
    one_key_record['enable'] = True
    one_key_record['command_file'] = record_cmd_path
    one_key_record['status_file'] = record_status_path
    one_key_record['queue_size'] = max(1024, int(one_key_record.get('queue_size', 0) or 0))
    one_key_record['fourcc'] = 'MJPG'
    one_key_record['video_fps'] = 40.0
    one_key_record['write_frame_index_csv'] = True
    one_key_record['recording_runtime_note'] = 'v02_10min_mjpg_queue1024'
    # ---- YSXQ one-key record path fix end ----

    # Ensure the event-side non-blocking human mask prefilter reads the same MVS
    # config that the MVS/YOLO process will actually use, and use the same
    # absolute human JSONL template to avoid cwd-dependent lookup.
    shared = event_cfg.setdefault('shared_options', {})
    shared['event_human_mask_mvs_config'] = str(mvs_path)
    shared['event_human_mask_jsonl_template'] = str(root_path.resolve() / 'human_result_{camera}.jsonl')
    shared['event_sync_source'] = 'mvs_soft_trigger'
    shared['event_sync_timeline_jsonl'] = str(root_path.resolve() / 'global_trigger_timeline.jsonl')
    shared['event_sync_map_jsonl_template'] = str(root_path.resolve() / 'event_trigger_map_{camera}.jsonl')
    shared['event_sync_health_json_template'] = str(root_path.resolve() / 'event_sync_health_{camera}.json')
    shared['event_sync_map_ring_json_template'] = str(root_path.resolve() / 'event_sync_map_ring_{camera}.json')
    shared['event_sync_map_ring_records'] = 512
    shared['event_sync_map_ring_write_interval_ms'] = 100.0
    _inject_event_sync_start_options(event_cfg, project_root)
    obs_port_changes = _normalize_event_obs_srt_ports(event_cfg)
    if obs_port_changes:
        print(
            f"[launcher] event OBS-SRT duplicate ports normalized: {', '.join(obs_port_changes)}",
            flush=True,
        )

    event_path.write_text(json.dumps(event_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    mvs_path.write_text(json.dumps(mvs_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[launcher] unified config={unified_path}", flush=True)
    print(f"[launcher] runtime event config={event_path}", flush=True)
    print(f"[launcher] runtime mvs config={mvs_path}", flush=True)
    return event_path, mvs_path, event_cfg



def _patch_event_runtime_cpu_options(event_cfg_path: Path, project_root: Path, args, cams: List[str]) -> None:
    """Inject event-process CPU optimization arguments into runtime event config.

    The event cameras are launched by multi_camera_launcher.py, which forwards
    shared_options to every metavision_spatter_tracking... child process.  Keep
    this patch after runtime config generation so command-line choices override
    JSON defaults without modifying the user's editable config.
    """
    try:
        cfg = load_json(event_cfg_path)
        shared = cfg.setdefault('shared_options', {})
        shared['event_empty_slice_fastpath_enable'] = bool(getattr(args, 'event_empty_slice_fastpath_enable', True))
        shared['event_small_slice_event_thresh'] = int(getattr(args, 'event_small_slice_event_thresh', 0) or 0)
        shared['event_human_mask_cache_enable'] = bool(getattr(args, 'event_human_mask_cache_enable', True))
        shared['event_human_mask_cache_max_fps'] = float(getattr(args, 'event_human_mask_cache_max_fps', 40.0) or 40.0)
        shared['event_human_mask_cache_stale_ms'] = float(getattr(args, 'event_human_mask_cache_stale_ms', 150.0) or 150.0)
        event_worker_human_mask_filter_enable = bool(getattr(args, 'event_worker_human_mask_filter_enable', True))
        shared['enable_event_human_mask_filter'] = event_worker_human_mask_filter_enable
        shared['event_human_mask_buffer_enable'] = bool(getattr(args, 'event_human_mask_buffer_enable', False))
        shared['event_human_mask_buffer_ms'] = float(getattr(args, 'event_human_mask_buffer_ms', 90.0) or 90.0)
        shared['event_human_mask_buffer_timeout_ms'] = float(getattr(args, 'event_human_mask_buffer_timeout_ms', 150.0) or 150.0)
        shared['event_human_mask_buffer_release_policy'] = str(getattr(args, 'event_human_mask_buffer_release_policy', 'raw_at_budget') or 'raw_at_budget')
        shared['event_human_mask_buffer_max_events'] = int(getattr(args, 'event_human_mask_buffer_max_events', 300000) or 300000)
        shared['event_human_mask_buffer_max_buckets'] = int(getattr(args, 'event_human_mask_buffer_max_buckets', 8) or 8)
        shared['event_human_mask_buffer_empty_release_fps'] = float(getattr(args, 'event_human_mask_buffer_empty_release_fps', 0.0) or 0.0)
        shared['event_human_mask_filter_max_fps'] = float(getattr(args, 'event_human_mask_filter_max_fps', 0.0) or 0.0)
        shared['event_human_mask_stats_jsonl'] = str(getattr(args, 'event_human_mask_stats_jsonl', '') or '')
        shared['event_human_mask_stats_max_fps'] = float(getattr(args, 'event_human_mask_stats_max_fps', 2.0) or 0.0)
        shared['event_human_mask_stats_flush_ms'] = float(getattr(args, 'event_human_mask_stats_flush_ms', 1000.0) or 0.0)
        shared['event_visual_mask_overlay_enable'] = bool(getattr(args, 'event_visual_mask_overlay_enable', False))
        shared['event_visual_mask_raster_template'] = str(getattr(args, 'event_visual_mask_raster_template', '/dev/shm/event_human_mask_raster_{camera}.mmap') or '/dev/shm/event_human_mask_raster_{camera}.mmap')
        shared['event_visual_mask_overlay_name_template'] = str(getattr(args, 'event_visual_mask_overlay_name_template', 'event_overlay_masked_latest_{camera}') or 'event_overlay_masked_latest_{camera}')
        shared['event_visual_mask_overlay_sidecar_template'] = str(getattr(args, 'event_visual_mask_overlay_sidecar_template', 'sync_ipc/event_overlay_masked_{camera}_latest.json') or 'sync_ipc/event_overlay_masked_{camera}_latest.json')
        shared['event_visual_mask_overlay_history_enable'] = bool(getattr(args, 'event_visual_mask_overlay_history_enable', False))
        shared['event_visual_mask_overlay_history_name_template'] = str(getattr(args, 'event_visual_mask_overlay_history_name_template', 'event_overlay_masked_hist_{camera}_{slot}') or 'event_overlay_masked_hist_{camera}_{slot}')
        shared['event_visual_mask_overlay_history_sidecar_template'] = str(getattr(args, 'event_visual_mask_overlay_history_sidecar_template', 'sync_ipc/event_overlay_masked_{camera}_history.json') or 'sync_ipc/event_overlay_masked_{camera}_history.json')
        shared['event_visual_mask_overlay_history_slots'] = int(getattr(args, 'event_visual_mask_overlay_history_slots', 64) or 64)
        shared['event_visual_mask_overlay_sparse_enable'] = bool(getattr(args, 'event_visual_mask_overlay_sparse_enable', False))
        shared['event_visual_mask_overlay_sparse_name_template'] = str(getattr(args, 'event_visual_mask_overlay_sparse_name_template', 'event_overlay_sparse_hist_{camera}_{slot}') or 'event_overlay_sparse_hist_{camera}_{slot}')
        shared['event_visual_mask_overlay_sparse_sidecar_template'] = str(getattr(args, 'event_visual_mask_overlay_sparse_sidecar_template', 'sync_ipc/event_overlay_sparse_{camera}_history.json') or 'sync_ipc/event_overlay_sparse_{camera}_history.json')
        shared['event_visual_mask_overlay_sparse_slots'] = int(getattr(args, 'event_visual_mask_overlay_sparse_slots', 64) or 64)
        shared['event_visual_mask_overlay_sparse_max_points'] = int(getattr(args, 'event_visual_mask_overlay_sparse_max_points', 20000) or 20000)
        shared['event_visual_mask_no_mask_policy'] = str(getattr(args, 'event_visual_mask_no_mask_policy', 'hold_last_good') or 'hold_last_good')
        shared['event_visual_mask_slice_us'] = int(getattr(args, 'event_visual_mask_slice_us', 33333) or 33333)
        shared['event_visual_mask_accumulation_us'] = int(getattr(args, 'event_visual_mask_accumulation_us', 33333) or 33333)
        shared['event_visual_mask_publish_fps'] = int(getattr(args, 'event_visual_mask_publish_fps', 40) or 40)
        shared['event_visual_mask_stale_ms'] = int(getattr(args, 'event_visual_mask_stale_ms', 120) or 120)
        shared['event_sync_source'] = str(getattr(args, 'event_sync_source', shared.get('event_sync_source', 'mvs_soft_trigger')) or 'mvs_soft_trigger')
        shared['event_sync_timeline_jsonl'] = str((project_root / 'sync_ipc' / 'global_trigger_timeline.jsonl').resolve())
        shared['event_sync_map_jsonl_template'] = str((project_root / 'sync_ipc' / 'event_trigger_map_{camera}.jsonl').resolve())
        shared['event_sync_health_json_template'] = str((project_root / 'sync_ipc' / 'event_sync_health_{camera}.json').resolve())
        shared['event_sync_map_ring_json_template'] = str((project_root / 'sync_ipc' / 'event_sync_map_ring_{camera}.json').resolve())
        shared['event_sync_map_ring_records'] = int(getattr(args, 'event_sync_map_ring_records', 512) or 512)
        shared['event_sync_map_ring_write_interval_ms'] = float(getattr(args, 'event_sync_map_ring_write_interval_ms', 100.0) or 100.0)
        shared['event_sync_wall_guard_ms'] = float(getattr(args, 'event_sync_wall_guard_ms', 20.0) or 20.0)
        shared['event_sync_offset_jump_warn_frames'] = int(getattr(args, 'event_sync_offset_jump_warn_frames', 2) or 2)
        shared['event_sync_local_fallback_enable'] = bool(getattr(args, 'event_sync_local_fallback_enable', True))
        shared['tsf1_pub_fps'] = float(getattr(args, 'tsf1_pub_fps', 8.0) or 8.0)
        shared['tsf1_pub_accumulation_time_us'] = int(getattr(args, 'tsf1_pub_accumulation_time_us', 12000) or 12000)
        shared['event_overlay_pub_max_fps'] = float(getattr(args, 'event_overlay_pub_max_fps', 170.0) or 0.0)
        shared['event_overlay_pub_channels'] = int(getattr(args, 'event_overlay_pub_channels', 3) or 3)
        shared['event_overlay_render_mode'] = str(getattr(args, 'event_overlay_render_mode', 'bgr') or 'bgr')
        shared['event_overlay_mono_color'] = str(getattr(args, 'event_overlay_mono_color', 'white') or 'white')
        shared['event_overlay_mono_decay_us'] = int(getattr(args, 'event_overlay_mono_decay_us', 0) or 0)
        if bool(getattr(args, 'event_overlay_draw_debug_text', True)):
            shared['event_overlay_draw_debug_text'] = True
            shared.pop('disable_event_overlay_draw_debug_text', None)
        else:
            shared.pop('event_overlay_draw_debug_text', None)
            shared['disable_event_overlay_draw_debug_text'] = True
        shared['event_tsf1_pub_max_fps'] = float(getattr(args, 'event_tsf1_pub_max_fps', 170.0) or 0.0)
        shared['event_obs_submit_max_fps'] = float(getattr(args, 'event_obs_submit_max_fps', 170.0) or 0.0)
        shared['event_viz_after_human_mask_filter'] = bool(getattr(args, 'event_viz_after_human_mask_filter', True))
        shared['event_viz_buffer_enable'] = bool(getattr(args, 'event_viz_buffer_enable', True))
        shared['event_viz_buffer_ms'] = float(getattr(args, 'event_viz_buffer_ms', 50.0) or 50.0)
        shared['event_viz_buffer_slots'] = int(getattr(args, 'event_viz_buffer_slots', 0) or 0)
        shared['event_viz_buffer_drop_policy'] = str(getattr(args, 'event_viz_buffer_drop_policy', 'drop_oldest') or 'drop_oldest')
        shared['event_viz_max_output_age_ms'] = float(getattr(args, 'event_viz_max_output_age_ms', 50.0) or 50.0)
        shared['event_viz_build_max_fps'] = float(getattr(args, 'event_viz_build_max_fps', 0.0) or 0.0)
        shared['event_viz_skip_empty_fastpath'] = bool(getattr(args, 'event_viz_skip_empty_fastpath', False))
        shared['event_viz_worker_mode'] = str(getattr(args, 'event_viz_worker_mode', 'thread') or 'thread')
        shared['event_viz_no_mask_policy'] = str(getattr(args, 'event_viz_no_mask_policy', 'empty') or 'empty')
        shared['event_viz_idle_heartbeat_fps'] = float(getattr(args, 'event_viz_idle_heartbeat_fps', 20.0) or 0.0)
        shared['event_track_input_policy'] = str(getattr(args, 'event_track_input_policy', 'post_mask') or 'post_mask')
        shared['event_viz_sidecar_template'] = str((project_root / 'sync_ipc' / 'event_viz_{camera}_latest.json').resolve())
        shared['event_overlay_sidecar_template'] = str((project_root / 'sync_ipc' / 'event_overlay_{camera}_latest.json').resolve())
        shared['event_obs_sidecar_enable'] = bool(getattr(args, 'event_obs_sidecar_enable', True))
        shared['event_obs_sidecar_template'] = str((project_root / 'sync_ipc' / 'event_obs_{camera}_latest.json').resolve())
        shared['event_overlay_shm_schema_version'] = int(getattr(args, 'event_overlay_shm_schema_version', 2) or 2)
        shared['event_worker_status_enable'] = bool(getattr(args, 'event_worker_status_enable', True))
        shared['event_worker_status_interval_ms'] = float(getattr(args, 'event_worker_status_interval_ms', 1000.0) or 1000.0)
        shared['event_worker_status_file'] = str((project_root / 'sync_ipc' / 'event_worker_{camera}_status.json').resolve())
        shared['event_hot_pixel_filter_enable'] = bool(getattr(args, 'event_hot_pixel_filter_enable', False))
        shared['event_hot_pixel_filter_mode'] = str(getattr(args, 'event_hot_pixel_filter_mode', 'tracking_only') or 'tracking_only')
        shared['event_hot_pixel_radius_px'] = int(getattr(args, 'event_hot_pixel_radius_px', 0) or 0)
        shared['event_hot_pixel_static_map'] = str(getattr(args, 'event_hot_pixel_static_map', '') or '')
        shared['event_hot_pixel_learn_enable'] = bool(getattr(args, 'event_hot_pixel_learn_enable', False))
        shared['event_hot_pixel_learn_min_fraction'] = float(getattr(args, 'event_hot_pixel_learn_min_fraction', 0.25) or 0.25)
        shared['event_hot_pixel_learn_min_count'] = int(getattr(args, 'event_hot_pixel_learn_min_count', 64) or 64)
        shared['event_hot_pixel_learn_confirm_slices'] = int(getattr(args, 'event_hot_pixel_learn_confirm_slices', 5) or 5)
        shared['event_hot_pixel_learn_max_pixels'] = int(getattr(args, 'event_hot_pixel_learn_max_pixels', 16) or 16)
        shared['event_cluster_sanitize_enable'] = bool(getattr(args, 'event_cluster_sanitize_enable', True))
        shared['event_line_shadow_enable'] = bool(getattr(args, 'event_line_shadow_enable', False))
        shared['event_line_shadow_cameras'] = str(getattr(args, 'event_line_shadow_cameras', '') or '')
        shared['event_line_shadow_min_step_px'] = float(getattr(args, 'event_line_shadow_min_step_px', -1.0) or -1.0)
        shared['event_line_shadow_min_total_disp_px'] = float(getattr(args, 'event_line_shadow_min_total_disp_px', -1.0) or -1.0)
        shared['event_line_shadow_bullet_min_output_streak'] = int(getattr(args, 'event_line_shadow_bullet_min_output_streak', -1) or -1)
        shared['event_line_shadow_untracked_threshold'] = int(getattr(args, 'event_line_shadow_untracked_threshold', -1) or -1)
        shared['event_line_shadow_line_min_valid_step_count'] = int(getattr(args, 'event_line_shadow_line_min_valid_step_count', -1) or -1)
        shared['event_sequential_start_ready_gate_enable'] = bool(getattr(args, 'event_sequential_start_ready_gate_enable', False))
        shared['event_sequential_start_timeout_sec'] = float(getattr(args, 'event_sequential_start_timeout_sec', 8.0) or 8.0)
        shared['event_sequential_start_poll_sec'] = float(getattr(args, 'event_sequential_start_poll_sec', 0.2) or 0.2)
        shared['event_sequential_start_settle_sec'] = float(getattr(args, 'event_sequential_start_settle_sec', 0.0) or 0.0)
        shared['event_sequential_start_continue_on_failure'] = bool(getattr(args, 'event_sequential_start_continue_on_failure', True))
        shared['event_loop_profiler_enable'] = bool(getattr(args, 'event_loop_profiler_enable', False))
        shared['event_loop_profiler_interval_ms'] = float(getattr(args, 'event_loop_profiler_interval_ms', 1000.0) or 1000.0)
        shared['event_loop_profiler_jsonl_template'] = str((project_root / 'sync_ipc' / 'event_loop_profile_{camera}.jsonl').resolve())
        shared['event_source_mode'] = str(getattr(args, 'event_source_mode', 'live_hal') or 'live_hal')
        shared['event_file_fifo_broker_bin'] = str(getattr(args, 'event_file_fifo_broker_bin', '') or '')
        shared['event_file_fifo_broker_raw_files'] = str(getattr(args, 'event_file_fifo_broker_raw_files', '') or '')
        shared['event_file_fifo_broker_replay_timing'] = str(getattr(args, 'event_file_fifo_broker_replay_timing', 'realtime') or 'realtime')
        shared['event_file_fifo_broker_allow_missing_ext_trigger'] = bool(getattr(args, 'event_file_fifo_broker_allow_missing_ext_trigger', False))
        shared['event_file_fifo_broker_loop'] = bool(getattr(args, 'event_file_fifo_broker_loop', False))
        shared['mvs_source_mode'] = str(getattr(args, 'mvs_source_mode', 'live_camera') or 'live_camera')
        shared['mvs_file_frame_broker_dataset_dir'] = str(getattr(args, 'mvs_file_frame_broker_dataset_dir', '') or '')
        shared['mvs_file_frame_broker_replay_timing'] = str(getattr(args, 'mvs_file_frame_broker_replay_timing', 'realtime') or 'realtime')
        shared['mvs_file_frame_broker_loop'] = bool(getattr(args, 'mvs_file_frame_broker_loop', False))
        shared['event_native_hal_broker_enable'] = bool(getattr(args, 'event_native_hal_broker_enable', False))
        shared['event_native_hal_broker_bin'] = str(getattr(args, 'event_native_hal_broker_bin', '') or '')
        shared['event_native_hal_broker_auto_build'] = bool(getattr(args, 'event_native_hal_broker_auto_build', True))
        shared['event_native_hal_broker_build_dir'] = str(getattr(args, 'event_native_hal_broker_build_dir', 'native_event_bridge/build') or 'native_event_bridge/build')
        shared['event_native_hal_broker_out_dir'] = str(getattr(args, 'event_native_hal_broker_out_dir', 'sync_ipc/native_hal_fifos') or 'sync_ipc/native_hal_fifos')
        shared['event_native_hal_broker_fifo_open_mode'] = str(getattr(args, 'event_native_hal_broker_fifo_open_mode', 'rdwr') or 'rdwr')
        shared['event_native_hal_broker_duration_sec'] = int(getattr(args, 'event_native_hal_broker_duration_sec', 0) or 0)
        shared['event_native_hal_broker_slice_us'] = int(getattr(args, 'event_native_hal_broker_slice_us', 4000) or 4000)
        shared['event_native_hal_broker_wait_mode'] = str(getattr(args, 'event_native_hal_broker_wait_mode', 'poll') or 'poll')
        shared['event_native_hal_broker_poll_sleep_us'] = int(getattr(args, 'event_native_hal_broker_poll_sleep_us', 100) or 0)
        shared['event_native_hal_broker_report_ms'] = int(getattr(args, 'event_native_hal_broker_report_ms', 1000) or 1000)
        shared['event_native_hal_broker_open_retries'] = int(getattr(args, 'event_native_hal_broker_open_retries', 1) or 1)
        shared['event_native_hal_broker_open_retry_delay_ms'] = int(getattr(args, 'event_native_hal_broker_open_retry_delay_ms', 0) or 0)
        shared['event_native_hal_broker_stats_jsonl'] = str(getattr(args, 'event_native_hal_broker_stats_jsonl', 'sync_ipc/event_native_hal_broker_stats.jsonl') or 'sync_ipc/event_native_hal_broker_stats.jsonl')
        shared['event_native_hal_broker_stdout_log'] = str(getattr(args, 'event_native_hal_broker_stdout_log', 'sync_ipc/launcher_logs/event_native_hal_broker.out') or 'sync_ipc/launcher_logs/event_native_hal_broker.out')
        shared['event_native_hal_broker_stderr_log'] = str(getattr(args, 'event_native_hal_broker_stderr_log', 'sync_ipc/launcher_logs/event_native_hal_broker.err') or 'sync_ipc/launcher_logs/event_native_hal_broker.err')
        shared['event_native_hal_broker_startup_timeout_sec'] = float(getattr(args, 'event_native_hal_broker_startup_timeout_sec', 30.0) or 30.0)
        shared['event_native_hal_broker_cpus'] = str(getattr(args, 'event_native_hal_broker_cpus', '') or '')
        marker_requested = bool(getattr(args, 'active_led_marker_enable', False))
        marker_cameras = {
            item.strip().upper()
            for item in str(getattr(args, 'active_led_marker_cameras', '') or '').replace(';', ',').split(',')
            if item.strip()
        }
        shared['active_led_marker_enable'] = False
        shared['active_led_marker_roi'] = str(getattr(args, 'active_led_marker_roi', '') or '')
        shared['active_led_marker_output'] = str((project_root / 'sync_ipc' / 'active_marker_observation_{camera}.jsonl').resolve())
        shared['active_led_marker_status_json'] = str((project_root / 'sync_ipc' / 'active_marker_status_{camera}.json').resolve())
        shared['active_led_marker_perf_csv'] = str((project_root / 'sync_ipc' / 'active_marker_perf_{camera}.csv').resolve())
        shared['active_led_marker_run_id'] = str(getattr(args, 'active_led_marker_run_id', '') or '')
        shared['active_led_marker_queue_size'] = int(getattr(args, 'active_led_marker_queue_size', 512) or 512)
        shared['active_led_marker_search_margin_px'] = int(getattr(args, 'active_led_marker_search_margin_px', 12) or 0)
        shared['active_led_marker_stale_ms'] = float(getattr(args, 'active_led_marker_stale_ms', 1000.0) or 1000.0)
        shared['active_led_marker_alpha_ms'] = float(getattr(args, 'active_led_marker_alpha_ms', 20.0) or 20.0)
        shared['active_led_marker_bit_count'] = int(getattr(args, 'active_led_marker_bit_count', 8) or 8)
        shared['active_led_marker_checksum'] = bool(getattr(args, 'active_led_marker_checksum', False))
        shared['active_led_marker_tolerance'] = float(getattr(args, 'active_led_marker_tolerance', 0.35) or 0.35)
        shared['active_led_marker_bin_us'] = int(getattr(args, 'active_led_marker_bin_us', 1000) or 1000)
        shared['active_led_marker_min_events_per_bin'] = int(getattr(args, 'active_led_marker_min_events_per_bin', 8) or 8)
        shared['active_led_marker_min_segment_events'] = int(getattr(args, 'active_led_marker_min_segment_events', 20) or 20)
        shared['active_led_marker_merge_gap_us'] = int(getattr(args, 'active_led_marker_merge_gap_us', 2500) or 2500)
        shared['active_led_marker_polarity'] = str(getattr(args, 'active_led_marker_polarity', 'any') or 'any')
        if str(getattr(args, 'event_worker_cpus', '') or '').strip():
            shared['event_worker_cpus'] = str(getattr(args, 'event_worker_cpus'))
        cameras_cfg = cfg.get('cameras', [])
        if isinstance(cameras_cfg, list):
            for cam_cfg in cameras_cfg:
                if not isinstance(cam_cfg, dict):
                    continue
                cam_cfg['enable_event_human_mask_filter'] = event_worker_human_mask_filter_enable
                alias = str(cam_cfg.get('alias') or cam_cfg.get('camera_alias') or '').strip().upper()
                cam_cfg['active_led_marker_enable'] = bool(marker_requested and alias in marker_cameras)
        obs_port_changes = _normalize_event_obs_srt_ports(cfg)
        event_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        if obs_port_changes:
            print(
                f"[launcher] event OBS-SRT duplicate ports normalized: {', '.join(obs_port_changes)}",
                flush=True,
            )
        print(
            f"[launcher] event CPU options patched: empty_fastpath={shared['event_empty_slice_fastpath_enable']} "
            f"human_mask_cache={shared['event_human_mask_cache_enable']} "
            f"overlay_max_fps={shared['event_overlay_pub_max_fps']} "
            f"overlay_channels={shared['event_overlay_pub_channels']} "
            f"overlay_render={shared['event_overlay_render_mode']} "
            f"tsf1_pub_fps={shared['tsf1_pub_fps']} "
            f"tsf1_accum_us={shared.get('tsf1_pub_accumulation_time_us')} "
            f"tsf1_max_fps={shared['event_tsf1_pub_max_fps']} "
            f"obs_max_fps={shared['event_obs_submit_max_fps']} "
            f"viz_build_max_fps={shared.get('event_viz_build_max_fps')} "
            f"viz_skip_empty={shared.get('event_viz_skip_empty_fastpath')} "
            f"mask_empty_release_fps={shared.get('event_human_mask_buffer_empty_release_fps')} "
            f"mask_stats_jsonl={'on' if shared.get('event_human_mask_stats_jsonl') else 'off'} "
            f"mask_stats_max_fps={shared.get('event_human_mask_stats_max_fps')} "
            f"event_sync_source={shared.get('event_sync_source')} "
            f"event_sync_wall_guard_ms={shared.get('event_sync_wall_guard_ms')} "
            f"viz_buffer_ms={shared.get('event_viz_buffer_ms')} "
            f"viz_worker={shared.get('event_viz_worker_mode')} "
            f"track_input={shared.get('event_track_input_policy')} "
            f"loop_profiler={shared.get('event_loop_profiler_enable')} "
            f"native_hal_broker={shared.get('event_native_hal_broker_enable')} "
            f"ir_id_marker={marker_requested}:{','.join(sorted(marker_cameras)) or 'off'} "
            f"mainloop_mask={shared.get('enable_event_human_mask_filter')} "
            f"event_worker_cpus={shared.get('event_worker_cpus','')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[launcher][WARN] patch event CPU options failed: {exc}", flush=True)

def parse_cams(args_cams: str, event_cfg: Dict) -> List[str]:
    if args_cams:
        return [x.strip() for x in args_cams.split(',') if x.strip()]
    cams = []
    for c in event_cfg.get('cameras', []):
        if c.get('alias') and c.get('enabled', True):
            cams.append(str(c['alias']).strip())
    return cams or list('ABCDE')


def _event_serial_map(event_cfg: Dict, cams: List[str]) -> Dict[str, str]:
    """Return camera alias -> event camera USB serial from the event config."""
    wanted = {str(c).strip(): "" for c in cams}
    for item in event_cfg.get('cameras', []):
        if not isinstance(item, dict):
            continue
        alias = str(item.get('alias') or item.get('camera') or item.get('camera_id') or item.get('cam') or item.get('name') or item.get('id') or '').strip()
        if alias not in wanted:
            continue
        serial = (
            item.get('serial')
            or item.get('serial_number')
            or item.get('device_serial')
            or item.get('event_serial')
            or item.get('metavision_serial')
            or item.get('serial_no')
            or ''
        )
        wanted[alias] = str(serial).strip()
    return wanted


def _ensure_default_mvs_env(log_cb: Optional[Callable[[str, str], None]] = None) -> None:
    """Provide Hikrobot MVS SDK defaults for non-interactive SSH launches."""
    if os.name == "nt":
        return
    lib_root = Path("/opt/MVS/lib")
    lib64 = lib_root / "64"
    camera_lib = lib64 / "libMvCameraControl.so"
    changed: List[str] = []
    if not os.environ.get("MVCAM_COMMON_RUNENV") and camera_lib.exists():
        os.environ["MVCAM_COMMON_RUNENV"] = str(lib_root)
        changed.append(f"MVCAM_COMMON_RUNENV={lib_root}")
    if lib64.exists():
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(":") if p]
        prepend = [str(lib64), str(lib_root)]
        new_parts = [p for p in prepend if p not in parts] + parts
        new_value = ":".join(new_parts)
        if new_value != current:
            os.environ["LD_LIBRARY_PATH"] = new_value
            changed.append("LD_LIBRARY_PATH+=/opt/MVS/lib/64")
    if changed and log_cb:
        log_cb("MVS SDK env defaulted for child processes: " + ", ".join(changed), "INFO")


def _json_read(path: Path) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8-sig') as fp:
            obj = json.load(fp)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _json_write_atomic(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
            fp.write('\n')
        os.replace(tmp, path)
    except Exception:
        pass


def _write_yolo_chain_mode_status(project_root: Path, args: argparse.Namespace, cams: List[str],
                                  legacy_enabled: bool, reason: str) -> None:
    """Write the authoritative runtime YOLO chain mode marker for diagnostics."""
    try:
        _json_write_atomic(project_root / 'sync_ipc' / 'yolo_chain_mode_status.json', {
            "schema_version": 1,
            "ts_wall_us": int(time.time() * 1_000_000),
            "global_yolo_enable": bool(getattr(args, 'global_yolo_enable', False)),
            "global_yolo_shadow_mode": bool(getattr(args, 'global_yolo_shadow_mode', False)),
            "global_yolo_primary_workers": int(getattr(args, 'global_yolo_primary_workers', 0) or 0),
            "global_yolo_primary_groups": str(getattr(args, 'global_yolo_primary_groups', '') or ''),
            "global_yolo_merge_enable": bool(getattr(args, 'global_yolo_merge_enable', False)),
            "legacy_yolo_shards_enabled": bool(legacy_enabled),
            "legacy_yolo_shards": str(getattr(args, 'mvs_yolo_shards', '') or ''),
            "legacy_yolo_shard_workers_enable": bool(getattr(args, 'mvs_yolo_shard_workers_enable', False)),
            "legacy_yolo_reason": str(reason or ''),
            "cams": list(cams),
        })
    except Exception:
        pass


def _recent_log_error(log_path: Path, cam: str, serial: str = "", max_lines: int = 500) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-max_lines:]
    except Exception:
        return ""
    needles = [str(cam).strip()]
    if serial:
        needles.append(str(serial).strip())
    important = ("LIBUSB_ERROR", "LibUSB", "RuntimeError", "Traceback", "Camera exception", "exception", "timeout")
    hits: List[str] = []
    for line in lines:
        low = line.lower()
        if any(n and n in line for n in needles) or any(k.lower() in low for k in important):
            if any(k.lower() in low for k in important):
                hits.append(line.strip())
    if hits:
        return hits[-1][-240:]
    return ""


def _run_usb_monitor(project_root: Path, serials: List[str], interval_sec: float, use_sudo: bool = False) -> Tuple[Dict[str, float], str]:
    serials = [str(s).strip() for s in serials if str(s).strip()]
    cached_rates, cached_error = _read_usb_monitor_cache(serials, max_age_sec=max(2.0, float(interval_sec or 0.5) * 4.0))
    if cached_rates:
        return cached_rates, cached_error
    script = project_root / 'monitor_camera_usb_throughput.py'
    if not script.exists():
        return {}, f"missing {script.name}"
    interval = max(0.1, float(interval_sec or 0.5))
    common = [
        str(script),
        '--json',
        '--once',
        '--interval',
        f'{interval:.3f}',
    ]
    if serials:
        common += ['--serials', ','.join(serials)]
    commands: List[List[str]] = [[sys.executable or 'python3'] + common]
    if bool(use_sudo) and os.name == 'posix':
        commands.append(['sudo', '-n', 'env', f'INTERVAL={interval:.3f}', sys.executable or 'python3'] + common)
    errors: List[str] = []
    for cmd in commands:
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(3.0, interval + 2.0),
            )
        except Exception as exc:
            errors.append(f"{Path(cmd[0]).name}: {type(exc).__name__}: {exc}")
            continue
        lines = [ln.strip() for ln in str(cp.stdout or '').splitlines() if ln.strip()]
        data: Dict[str, Any] = {}
        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                data = obj
                break
        if data:
            out: Dict[str, float] = {}
            ser_map = data.get('serials', {})
            if isinstance(ser_map, dict):
                for serial, rec in ser_map.items():
                    if isinstance(rec, dict):
                        val = rec.get('mbps', rec.get('total_MBps', rec.get('MBps', rec.get('mb_per_sec', 0.0))))
                    else:
                        val = rec
                    try:
                        out[str(serial)] = float(val)
                    except Exception:
                        out[str(serial)] = 0.0
            for key in ("active_count", "detected_count", "usb2_bad", "usb3_ok"):
                try:
                    out[f"__{key}__"] = float(data.get(key, 0.0) or 0.0)
                except Exception:
                    out[f"__{key}__"] = 0.0
            if out:
                err = str(data.get('error', '') or '')
                if cp.returncode not in (0, None) and not err:
                    err = f"monitor rc={cp.returncode}"
                return out, err
            errors.append("monitor returned no serial rates")
            continue
        err = (str(cp.stderr or '').strip() or str(cp.stdout or '').strip() or f"rc={cp.returncode}")[-300:]
        errors.append(f"{Path(cmd[0]).name}: {err}")
    return {}, "; ".join(errors[-2:])


def _usb_payload_to_rates(data: Dict[str, Any], serials: List[str]) -> Tuple[Dict[str, float], str]:
    out: Dict[str, float] = {}
    ser_map = data.get('serials', {})
    if isinstance(ser_map, dict):
        if serials:
            iterable = [(s, ser_map.get(s, {})) for s in serials]
        else:
            iterable = list(ser_map.items())
        for serial, rec in iterable:
            if isinstance(rec, dict):
                val = rec.get('mbps', rec.get('total_MBps', rec.get('MBps', rec.get('mb_per_sec', 0.0))))
            else:
                val = rec
            try:
                out[str(serial)] = float(val)
            except Exception:
                out[str(serial)] = 0.0
    for key in ("active_count", "detected_count", "usb2_bad", "usb3_ok"):
        try:
            out[f"__{key}__"] = float(data.get(key, 0.0) or 0.0)
        except Exception:
            out[f"__{key}__"] = 0.0
    err = str(data.get('error', '') or '')
    return out, err


def _read_usb_monitor_cache(serials: List[str], max_age_sec: float = 2.5) -> Tuple[Dict[str, float], str]:
    candidates = [
        Path('/dev/shm/ids_usb_camera_throughput.json'),
        Path('/tmp/ids_usb_camera_throughput.json'),
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            age = time.time() - float(path.stat().st_mtime)
            if age > max(0.5, float(max_age_sec or 2.5)):
                continue
            data = _json_read(path)
            if not data:
                continue
            rates, err = _usb_payload_to_rates(data, serials)
            if rates:
                return rates, f"cached:{path} age={age:.1f}s" + (f"; {err}" if err else "")
        except Exception:
            continue
    return {}, ""


def _write_startup_diag(
    diag_path: Optional[Path],
    *,
    state: str,
    title: str,
    cams: List[str],
    statuses: Dict[str, Dict[str, Any]],
    threshold_mbps: float,
    message: str = "",
    usb_error: str = "",
    started_wall: float = 0.0,
) -> None:
    if diag_path is None:
        return
    now = time.time()
    ready = [c for c in cams if bool(statuses.get(c, {}).get('ready', False))]
    missing = [c for c in cams if c not in ready]
    _json_write_atomic(diag_path, {
        "schema_version": 1,
        "kind": "event_startup_diagnostics",
        "state": str(state),
        "title": str(title),
        "message": str(message),
        "timestamp_wall": now,
        "age_since_event_start_sec": max(0.0, now - float(started_wall or now)),
        "cams": list(cams),
        "ready_cams": ready,
        "missing_cams": missing,
        "usb_threshold_mbps": float(threshold_mbps),
        "usb_error": str(usb_error or ""),
        "camera_status": statuses,
    })


def _start_startup_diag_window(project_root: Path, py: str, diag_path: Path, args: argparse.Namespace, log_cb: Callable[[str, str], None]) -> Optional[subprocess.Popen]:
    if not bool(getattr(args, 'event_startup_diagnostics_enable', True)):
        return None
    if not bool(getattr(args, 'event_startup_diagnostics_window_enable', True)):
        return None
    script = project_root / 'startup_diagnostics_window.py'
    if not script.exists():
        log_cb(f"startup diagnostics window script missing: {script}", "WARN")
        return None
    log_path = project_root / str(getattr(args, 'child_log_dir', 'sync_ipc/launcher_logs')) / 'startup_diagnostics.log'
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fp = open(log_path, 'a', encoding='utf-8', buffering=1)
        cmd = [
            py,
            str(script),
            '--json',
            str(diag_path),
            '--auto-exit-ok-ms',
            str(int(getattr(args, 'event_startup_diagnostics_auto_exit_ok_ms', 2500) or 2500)),
        ]
        proc = subprocess.Popen(cmd, cwd=str(project_root), stdout=fp, stderr=subprocess.STDOUT, start_new_session=True)
        log_cb(f"startup diagnostics window pid={proc.pid} json={diag_path}", "INFO")
        return proc
    except Exception as exc:
        log_cb(f"startup diagnostics window failed: {type(exc).__name__}: {exc}", "WARN")
        return None


def wait_ready(
    root: Path,
    cams: List[str],
    timeout: float,
    not_before_wall: float = 0.0,
    *,
    project_root: Optional[Path] = None,
    serial_map: Optional[Dict[str, str]] = None,
    usb_enable: bool = True,
    usb_threshold_mbps: float = 0.3,
    usb_interval_sec: float = 0.5,
    usb_sudo_enable: bool = False,
    allow_partial: bool = True,
    min_ready_cams: int = 1,
    partial_wait_sec: float = 8.0,
    diag_path: Optional[Path] = None,
    log_cb: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """Wait until every event camera is ready.

    A camera is accepted when it has a fresh ready flag or, if enabled, its USB
    stream throughput is above the configured MB/s threshold.  The USB branch
    catches the real field failure mode where a camera process starts but the
    device never produces events.
    """
    deadline = time.time() + float(timeout)
    missing: List[str] = []
    statuses: Dict[str, Dict[str, Any]] = {c: {} for c in cams}
    last_report = 0.0
    last_usb_sample = 0.0
    usb_rates: Dict[str, float] = {}
    usb_error = ""
    serial_map = serial_map or {}
    log = log_cb or (lambda msg, level="INFO": print(f"[launcher][{level}] {msg}", flush=True))
    while time.time() < deadline:
        now = time.time()
        if usb_enable and project_root is not None and (now - last_usb_sample >= max(0.2, float(usb_interval_sec or 0.5))):
            usb_rates, usb_error = _run_usb_monitor(
                project_root,
                [serial_map.get(c, "") for c in cams],
                usb_interval_sec,
                use_sudo=bool(usb_sudo_enable),
            )
            last_usb_sample = now
        missing = []
        for c in cams:
            status: Dict[str, Any] = {
                "ready": False,
                "ready_by": "",
                "flag_ready": False,
                "flag_mtime": 0.0,
                "usb_ready": False,
                "usb_mbps": 0.0,
                "serial": str(serial_map.get(c, "") or ""),
                "reason": "",
            }
            flag = root / f'event_ready_{c}.flag'
            if flag.exists():
                try:
                    mtime = float(flag.stat().st_mtime)
                    status["flag_mtime"] = mtime
                    if float(not_before_wall or 0.0) <= 0.0 or mtime >= float(not_before_wall) - 0.001:
                        status["flag_ready"] = True
                        status["ready"] = True
                        status["ready_by"] = "ready_flag"
                except Exception:
                    status["reason"] = "ready flag stat failed"
            else:
                status["reason"] = "ready flag missing"
            serial = str(serial_map.get(c, "") or "")
            if usb_enable and serial:
                rate = float(usb_rates.get(serial, 0.0) or 0.0)
                status["usb_mbps"] = rate
                if rate >= float(usb_threshold_mbps or 0.3):
                    status["usb_ready"] = True
                    status["ready"] = True
                    status["ready_by"] = status["ready_by"] or "usb_throughput"
            worker_status_path = root / f'event_worker_{c}_status.json'
            worker_status = _json_read(worker_status_path)
            if worker_status:
                status["worker_status_mtime"] = float(worker_status_path.stat().st_mtime) if worker_status_path.exists() else 0.0
                for key in ("state", "error", "last_error", "event_rate_mevps", "raw_event_count", "filtered_event_count", "published_fps"):
                    if key in worker_status:
                        status[f"worker_{key}"] = worker_status.get(key)
            if not status["ready"]:
                if usb_enable and serial and usb_error:
                    status["reason"] = status["reason"] or usb_error
                recent_error = _recent_log_error(root / 'launcher_logs' / 'event.log', c, serial)
                if recent_error:
                    status["recent_event_log_error"] = recent_error
                    status["reason"] = recent_error
                missing.append(c)
            statuses[c] = status
        usb_active_count = int(float(usb_rates.get("__active_count__", 0.0) or 0.0))
        usb_detected_count = int(float(usb_rates.get("__detected_count__", 0.0) or 0.0))
        has_serial_mapping = any(str(serial_map.get(c, "") or "").strip() for c in cams)
        if usb_enable and not has_serial_mapping and usb_active_count >= len(cams) and len(cams) > 0:
            missing = []
            for c in cams:
                statuses[c]["ready"] = True
                statuses[c]["ready_by"] = statuses[c].get("ready_by") or "usb_aggregate"
                statuses[c]["usb_active_count"] = usb_active_count
                statuses[c]["usb_detected_count"] = usb_detected_count
        ready_count = sum(1 for c in cams if bool(statuses.get(c, {}).get("ready", False)))
        effective_ready_count = max(ready_count, usb_active_count if not has_serial_mapping else ready_count)
        if not missing:
            _write_startup_diag(
                diag_path,
                state="ok",
                title="EVENT cameras ready",
                cams=cams,
                statuses=statuses,
                threshold_mbps=usb_threshold_mbps,
                message="all event cameras passed ready gate",
                usb_error=usb_error,
                started_wall=not_before_wall,
            )
            log(f"event ready OK: {','.join(cams)}", "INFO")
            return True
        elapsed = max(0.0, now - float(not_before_wall or now))
        min_needed = max(0, min(len(cams), int(min_ready_cams or 0)))
        if bool(allow_partial) and elapsed >= max(0.0, float(partial_wait_sec or 0.0)) and effective_ready_count >= min_needed:
            _write_startup_diag(
                diag_path,
                state="degraded",
                title="EVENT startup degraded; continuing",
                cams=cams,
                statuses=statuses,
                threshold_mbps=usb_threshold_mbps,
                message=(
                    f"continuing with ready={ready_count}/{len(cams)} effective_ready={effective_ready_count}; "
                    f"missing={','.join(missing)}; usb_active={usb_rates.get('__active_count__', 0):.0f} "
                    f"detected={usb_rates.get('__detected_count__', 0):.0f}"
                ),
                usb_error=usb_error,
                started_wall=not_before_wall,
            )
            log(
                f"event startup degraded but tolerated: ready={ready_count}/{len(cams)} effective_ready={effective_ready_count} "
                f"missing={','.join(missing)} usb_active={usb_rates.get('__active_count__', 0):.0f}",
                "WARN",
            )
            return True
        if now - last_report >= 5.0:
            remain = max(0.0, deadline - now)
            detail = "; ".join(
                f"{c}:usb={statuses.get(c, {}).get('usb_mbps', 0.0):.3f}MB/s reason={statuses.get(c, {}).get('reason', '')}"
                for c in missing
            )
            log(f"waiting event ready; missing={','.join(missing)} timeout_in={remain:.1f}s {detail}", "WARN")
            _write_startup_diag(
                diag_path,
                state="starting",
                title="Waiting for EVENT cameras",
                cams=cams,
                statuses=statuses,
                threshold_mbps=usb_threshold_mbps,
                message=f"missing/stale={','.join(missing)} timeout_in={remain:.1f}s",
                usb_error=usb_error,
                started_wall=not_before_wall,
            )
            last_report = now
        time.sleep(0.2)
    ready_count = sum(1 for c in cams if bool(statuses.get(c, {}).get("ready", False)))
    usb_active_count = int(float(usb_rates.get("__active_count__", 0.0) or 0.0))
    has_serial_mapping = any(str(serial_map.get(c, "") or "").strip() for c in cams)
    effective_ready_count = max(ready_count, usb_active_count if not has_serial_mapping else ready_count)
    min_needed = max(0, min(len(cams), int(min_ready_cams or 0)))
    if bool(allow_partial) and effective_ready_count >= min_needed:
        _write_startup_diag(
            diag_path,
            state="degraded",
            title="EVENT ready timeout tolerated",
            cams=cams,
            statuses=statuses,
            threshold_mbps=usb_threshold_mbps,
            message=f"timeout tolerated with ready={ready_count}/{len(cams)} effective_ready={effective_ready_count}; missing/stale={','.join(missing)}",
            usb_error=usb_error,
            started_wall=not_before_wall,
        )
        log(f"event ready timeout tolerated; ready={ready_count}/{len(cams)} effective_ready={effective_ready_count} missing/stale={','.join(missing)}", "WARN")
        return True
    _write_startup_diag(
        diag_path,
        state="error",
        title="EVENT startup interrupted",
        cams=cams,
        statuses=statuses,
        threshold_mbps=usb_threshold_mbps,
        message=f"event ready timeout; missing/stale={','.join(missing)}",
        usb_error=usb_error,
        started_wall=not_before_wall,
    )
    log(f"event ready timeout; missing/stale={','.join(missing)}", "ERROR")
    return False


def cleanup_event_start_sync_files(project_root: Path, cams: List[str]) -> None:
    """Remove stale event readiness/trigger files before launching EVENT.

    The MVS trigger coordinator starts tick_id from zero.  If stale ready flags
    or trigger CSVs survive from an older run, event sync_index can start from a
    different origin than MVS program_sync_index.  Cleaning these files makes the
    first coordinated trigger of this run become the shared index origin.
    """
    root = project_root / 'sync_ipc'
    root.mkdir(parents=True, exist_ok=True)
    for name in ('sync_start.flag', 'global_trigger_timeline.jsonl', 'global_trigger_timeline_latest.json'):
        path = root / name
        if path.exists():
            try:
                path.unlink()
                print(f"[launcher] removed stale event sync file {path}", flush=True)
            except Exception as exc:
                print(f"[launcher][WARN] cannot remove stale event sync file {path}: {exc}", flush=True)
    for c in cams:
        for name in (
            f'event_ready_{c}.flag',
            f'event_trigger_{c}.csv',
            f'event_trigger_map_{c}.jsonl',
            f'event_sync_health_{c}.json',
            f'event_sync_map_ring_{c}.json',
            f'mvs_frame_audit_{c}.csv',
        ):
            path = root / name
            if path.exists():
                try:
                    path.unlink()
                    print(f"[launcher] removed stale event sync file {path}", flush=True)
                except Exception as exc:
                    print(f"[launcher][WARN] cannot remove stale event sync file {path}: {exc}", flush=True)


def cleanup_stale(project_root: Path, cams: List[str], clean_logs: bool = False) -> None:
    for prefix in ['mvs_latest_', 'event_overlay_latest_', 'event_overlay_masked_latest_', 'event_overlay_masked_hist_', 'event_overlay_sparse_hist_', 'event_human_mask_raster_']:
        for c in cams:
            for p in Path('/dev/shm').glob(f'{prefix}{c}*'):
                try:
                    p.unlink()
                    print(f"[launcher] removed stale {p}", flush=True)
                except Exception as exc:
                    print(f"[launcher][WARN] cannot remove {p}: {exc}", flush=True)
    for c in cams:
        for pat in [
            f'event_overlay_masked_{c}_latest.json',
            f'event_overlay_masked_{c}_history.json',
            f'event_overlay_sparse_{c}_history.json',
            f'event_human_mask_raster_{c}_latest.json',
            f'event_sync_map_ring_{c}.json',
        ]:
            p = project_root / 'sync_ipc' / pat
            if p.exists():
                try:
                    p.unlink()
                    print(f"[launcher] removed stale {p}", flush=True)
                except Exception as exc:
                    print(f"[launcher][WARN] cannot remove {p}: {exc}", flush=True)
    for pat in ['record_control.json', 'record_status.json', 'record_status_*.json']:
        for p in (project_root / 'sync_ipc').glob(pat):
            if p.exists():
                try:
                    p.unlink()
                    print(f"[launcher] removed {p}", flush=True)
                except Exception:
                    pass
    if clean_logs:
        for pat in ['human_result_*.jsonl','hit_candidate_*.jsonl','pseudo_hit_*.jsonl','pseudo_hit_all.jsonl','hit_judge_debug_*.jsonl','hit_reject_debug_*.jsonl','overlay_bullet_point_*.jsonl','overlay_bullet_point_all.jsonl','overlay_bullet_event_*.jsonl','overlay_bullet_event_all.jsonl','mvs_frame_audit_*.csv','frame_bundle_audit_*.csv']:
            for p in (project_root/'sync_ipc').glob(pat):
                try:
                    p.unlink()
                except Exception:
                    pass


def initialize_event_mask_control_defaults(project_root: Path, args: argparse.Namespace) -> None:
    """Reset event-mask hot controls to the active launch profile at process start.

    Status-window hot updates are intentionally runtime-only.  Without this reset,
    a stale event_mask_control.json from an earlier run can silently override the
    profile default, for example forcing the V23-B 30 ms mask ticker back to 4 ms.
    """
    if not (
        bool(getattr(args, 'event_mask_evidence_enable', False))
        or bool(getattr(args, 'event_visual_mask_overlay_enable', False))
    ):
        return
    raw_path = str(getattr(args, 'event_mask_evidence_control_json', 'sync_ipc/event_mask_control.json') or 'sync_ipc/event_mask_control.json')
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    payload = {
        "schema_version": 1,
        "seq": int(time.time() * 1000),
        "source": "launch_profile_defaults",
        "updated_wall_us": int(time.time() * 1_000_000),
        "event_human_mask_evidence_enable": True,
        "event_human_mask_evidence_period_ms": float(getattr(args, 'event_mask_evidence_period_ms', 30.0) or 30.0),
        "event_human_mask_evidence_stale_ms": float(getattr(args, 'event_mask_evidence_stale_ms', 120.0) or 120.0),
        "mask_raster_enable": bool(getattr(args, 'event_mask_raster_enable', False)),
        "mask_raster_dilate_px": int(getattr(args, 'event_mask_raster_dilate_px', 8) or 0),
        "mask_raster_erode_px": int(getattr(args, 'event_mask_raster_erode_px', 0) or 0),
        "mask_raster_min_conf": float(getattr(args, 'event_mask_raster_min_conf', 0.2) or 0.0),
        "event_visual_mask_slice_us": int(getattr(args, 'event_visual_mask_slice_us', 33333) or 33333),
        "event_visual_mask_accumulation_us": int(getattr(args, 'event_visual_mask_accumulation_us', 33333) or 33333),
        "event_visual_mask_slice_ms": round(float(getattr(args, 'event_visual_mask_slice_us', 33333) or 33333) / 1000.0, 3),
        "event_visual_mask_accumulation_ms": round(float(getattr(args, 'event_visual_mask_accumulation_us', 33333) or 33333) / 1000.0, 3),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding='utf-8')
        os.replace(tmp, path)
        print(f"[launcher] initialized event mask control defaults {path}", flush=True)
    except Exception as exc:
        print(f"[launcher][WARN] failed to initialize event mask control defaults {path}: {exc}", flush=True)


def resolve_script(project_root: Path, candidates: List[str]) -> str:
    """Return the first existing script path relative to project_root.

    This keeps the launcher compatible with both older flat layouts and the
    newer cpp_bullet_core/rentijiance layout.
    """
    for rel in candidates:
        if (project_root / rel).exists():
            return rel
    # Keep the first candidate so the child process prints a clear file-not-found error.
    print(
        f"[launcher][WARN] none of candidate scripts exists, using first anyway: {candidates}",
        flush=True,
    )
    return candidates[0]



def _taskset_cmd(cpu: str, cmd: List[str]) -> List[str]:
    cpu = str(cpu or "").strip()
    if not cpu:
        return cmd
    return ["taskset", "-c", cpu] + list(cmd)


def _split_worker_cpu(cpus: str, idx: int) -> str:
    items = [x.strip() for x in str(cpus or "").split(',') if x.strip()]
    if not items:
        return ""
    return items[min(idx, len(items) - 1)]


def _enable_external_mvs_shm_mode(mvs_cfg_path: Path, project_root: Path) -> None:
    """Patch runtime MVS config for split-worker architecture.

    In this mode the per-camera worker processes own the MVS SDK handles and publish
    /dev/shm/mvs_latest_{camera}.  The main MVS/YOLO process must not open cameras;
    it only reads these shared frames and keeps the YOLO/fusion/human JSON pipeline.
    """
    try:
        cfg = load_json(mvs_cfg_path)
        runtime = cfg.setdefault('runtime', {})
        runtime['external_mvs_shm_enable'] = True
        # Keep the OpenGL publisher names enabled and stable for workers and renderer.
        sync = cfg.setdefault('sync', {})
        sync['mvs_frame_pub_enable'] = True
        if not sync.get('mvs_frame_pub_name_template'):
            sync['mvs_frame_pub_name_template'] = 'mvs_latest_{camera}'
        root_raw = str(sync.get('root', './sync_ipc') or './sync_ipc')
        root_path = Path(root_raw).expanduser()
        if not root_path.is_absolute():
            root_path = project_root / root_path
        sync['root'] = str(root_path.resolve())
        mvs_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[launcher] split-worker mode patched runtime MVS config: external_mvs_shm_enable=true -> {mvs_cfg_path}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"failed to patch MVS runtime config for split workers: {exc}") from exc


DEFAULT_YOLO_SHARD_LAYOUT: List[List[str]] = [
    ["A", "C"],
    ["B", "D"],
    ["E", "G"],
    ["F", "H"],
]


def _parse_yolo_shards(shards: str, cams: List[str]) -> List[List[str]]:
    """Parse A,C:B,D:E,G:F,H into YOLO shard camera lists."""
    cams = [str(c).strip() for c in cams if str(c).strip()]
    raw = str(shards or '').strip()
    if not raw:
        valid = set(cams)
        out = [[c for c in group if c in valid] for group in DEFAULT_YOLO_SHARD_LAYOUT]
        out = [group for group in out if group]
        used = {c for group in out for c in group}
        missing = [c for c in cams if c not in used]
        if missing:
            out.append(missing)
        return out or [cams]
    out: List[List[str]] = []
    valid = set(cams)
    for group in raw.split(':'):
        items = [x.strip() for x in group.split(',') if x.strip()]
        items = [x for x in items if x in valid]
        if items:
            out.append(items)
    used = {c for group in out for c in group}
    missing = [c for c in cams if c not in used]
    if missing:
        out.append(missing)
    return out or [cams]


def _safe_worker_id_for_cams(cams: List[str]) -> str:
    return 'yolo_' + ''.join(str(c).strip() for c in cams if str(c).strip()).lower()


def _display_name_for_yolo_worker(shard_id: str) -> str:
    sid = str(shard_id or "").strip()
    if sid.lower().startswith("yolo_"):
        return sid.upper()
    return "YOLO_" + sid.upper()


def _write_yolo_shard_config(base_mvs_config: str, project_root: Path, shard_id: str, shard_cams: List[str], args: argparse.Namespace) -> Path:
    src = Path(base_mvs_config)
    if not src.is_absolute():
        src = project_root / src
    cfg = load_json(src)
    runtime = cfg.setdefault('runtime', {})
    sync = cfg.setdefault('sync', {})

    runtime['external_mvs_shm_enable'] = True
    runtime['yolo_worker_id'] = str(shard_id)
    runtime['yolo_worker_cams'] = list(shard_cams)
    runtime['external_mvs_shm_allowed_cams'] = list(shard_cams)
    runtime['ue5_headless_enable'] = True
    runtime['display_raw_fusion_every_mvs_frame'] = False
    runtime['yolo_stage_metrics_enable'] = True
    runtime['yolo_worker_status_interval_ms'] = float(getattr(args, 'yolo_worker_status_interval_ms', 500.0) or 500.0)
    if getattr(args, 'yolo_async_jsonl_writer_enable', False):
        runtime['yolo_async_jsonl_writer_enable'] = True
        runtime['yolo_async_jsonl_queue'] = int(getattr(args, 'yolo_async_jsonl_queue', 512) or 512)
        runtime['yolo_async_jsonl_flush_ms'] = float(getattr(args, 'yolo_async_jsonl_flush_ms', 50.0) or 50.0)
    if int(getattr(args, 'yolo_postprocess_workers', 0) or 0) > 0:
        runtime['parallel_postprocess_enable'] = True
        runtime['parallel_postprocess_workers'] = int(getattr(args, 'yolo_postprocess_workers'))
    if getattr(args, 'yolo_external_postprocess_enable', False):
        runtime['yolo_external_postprocess_enable'] = True
        runtime['parallel_postprocess_enable'] = False
        runtime['yolo_external_postprocess_workers'] = max(1, int(getattr(args, 'yolo_external_postprocess_workers', 1) or 1))
        runtime['yolo_external_postprocess_queue_size'] = max(1, int(getattr(args, 'yolo_external_postprocess_queue_size', 4) or 4))
        runtime['yolo_external_postprocess_cpus'] = _split_worker_cpu(str(getattr(args, 'yolo_external_postprocess_cpus', '') or ''), int(getattr(args, '_current_yolo_shard_index', 0) or 0))
        if str(getattr(args, 'yolo_external_postprocess_start_method', '') or '').strip():
            runtime['yolo_external_postprocess_start_method'] = str(getattr(args, 'yolo_external_postprocess_start_method')).strip()
    if float(getattr(args, 'yolo_drop_stale_input_ms', 0.0) or 0.0) > 0:
        runtime['yolo_drop_stale_input_ms'] = float(getattr(args, 'yolo_drop_stale_input_ms'))
    if float(getattr(args, 'yolo_drop_stale_result_ms', 0.0) or 0.0) > 0:
        runtime['yolo_drop_stale_result_ms'] = float(getattr(args, 'yolo_drop_stale_result_ms'))

    # Avoid multiple workers appending to one perf CSV.
    sync['yolo_infer_perf_csv_path'] = '{root}/yolo_infer_perf_' + str(shard_id) + '.csv'
    sync['truncate_yolo_infer_perf_on_start'] = True

    out_dir = project_root / 'sync_ipc' / '_runtime_config'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'mvs_runtime_{shard_id}.json'
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[launcher] YOLO shard config {shard_id}: cams={','.join(shard_cams)} -> {out}", flush=True)
    return out


def _env_limited_python_cmd(cmd: List[str], extra_env: Optional[Dict[str, str]] = None, cpu_threads: int = 1) -> List[str]:
    n = str(max(1, int(cpu_threads or 1)))
    pairs = {
        'OMP_NUM_THREADS': n,
        'MKL_NUM_THREADS': n,
        'OPENBLAS_NUM_THREADS': n,
        'NUMEXPR_NUM_THREADS': n,
    }
    if extra_env:
        pairs.update({str(k): str(v) for k, v in extra_env.items()})
    env_args: List[str] = ['env']
    for k, v in pairs.items():
        env_args.append(f'{k}={v}')
    return env_args + list(cmd)


def _cpu_hotspot_collect_cmd(project_root: Path, duration_sec: int, delay_sec: float, prefix: str) -> List[str]:
    root_q = shlex.quote(str(Path(project_root)))
    prefix_q = shlex.quote(str(prefix or 'cpu_hotspots'))
    duration = max(1, int(duration_sec or 60))
    delay = max(0.0, float(delay_sec or 0.0))
    script = f"""set -u
cd {root_q}
DUR={duration}
DELAY={delay}
PREFIX={prefix_q}
if [ "$DELAY" != "0" ] && [ "$DELAY" != "0.0" ]; then
  echo "[collect] waiting $DELAY sec before hotspot collection"
  sleep "$DELAY"
fi
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${{PREFIX}}_${{TS}}"
mkdir -p "$OUT" "$OUT/threads" "$OUT/status" "$OUT/logs"
echo "[collect] $OUT duration=${{DUR}}s"

{{
  echo "===== date ====="; date; echo
  echo "===== uptime ====="; uptime; echo
  echo "===== nproc ====="; nproc; echo
  echo "===== top processes initial ====="
  ps -eo pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -80
}} > "$OUT/top_processes_initial.txt" 2>&1

{{
  echo "===== relevant processes initial ====="
  pgrep -af "launch_fusion_system|global_yolo_infer_server|global_yolo_merge_publisher|global_person_id_server|person_id_dataset_recorder|event_mask_evidence_service|global_bullet_reprocess_service|global_bullet_fusion_service|damage_attribution_service|foam_dart_speed_monitor_service|manual_hit_service|game_event_bridge_service|global_bev_fusion_renderer|event_overlay_overview_renderer|pyopengl_cuda_preview_renderer|mvs_cam_worker|mvs_trigger_coordinator|hik_rgb_yolo|hit_judge|multi_camera_launcher|metavision|python" || true
}} > "$OUT/process_map_initial.txt" 2>&1

pgrep -f "launch_fusion_system|global_yolo_infer_server|global_yolo_merge_publisher|global_person_id_server|person_id_dataset_recorder|event_mask_evidence_service|global_bullet_reprocess_service|global_bullet_fusion_service|damage_attribution_service|foam_dart_speed_monitor_service|manual_hit_service|game_event_bridge_service|global_bev_fusion_renderer|event_overlay_overview_renderer|pyopengl_cuda_preview_renderer|mvs_cam_worker|mvs_trigger_coordinator|hik_rgb_yolo|hit_judge|multi_camera_launcher|metavision" | while read -r pid; do
  ps -p "$pid" -o pid,ppid,psr,pcpu,pmem,comm,args > "$OUT/threads/${{pid}}_process_initial.txt" 2>&1 || true
  ps -L -p "$pid" -o pid,tid,psr,pcpu,pmem,comm --sort=-pcpu > "$OUT/threads/${{pid}}_threads_initial.txt" 2>&1 || true
done

if command -v pidstat >/dev/null 2>&1; then
  {{
    echo "===== pidstat process samples ====="
    pidstat -p ALL -u -r -d 1 "$DUR"
  }} > "$OUT/pidstat_process.txt" 2>&1 &
  PIDSTAT_PROC_PID=$!
  {{
    echo "===== pidstat relevant thread samples ====="
    PIDS="$(pgrep -d, -f 'launch_fusion_system|global_yolo_infer_server|global_yolo_merge_publisher|global_person_id_server|person_id_dataset_recorder|event_mask_evidence_service|global_bullet_reprocess_service|global_bullet_fusion_service|damage_attribution_service|foam_dart_speed_monitor_service|manual_hit_service|game_event_bridge_service|global_bev_fusion_renderer|event_overlay_overview_renderer|pyopengl_cuda_preview_renderer|mvs_cam_worker|mvs_trigger_coordinator|hik_rgb_yolo|hit_judge|multi_camera_launcher|metavision' || true)"
    if [ -n "$PIDS" ]; then
      pidstat -t -p "$PIDS" -u -r 1 "$DUR"
    else
      echo "no relevant pids"
      sleep "$DUR"
    fi
  }} > "$OUT/pidstat_threads.txt" 2>&1 &
  PIDSTAT_THREAD_PID=$!
else
  {{
    echo "pidstat not installed"
    sleep "$DUR"
  }} > "$OUT/pidstat_process.txt" 2>&1 &
  PIDSTAT_PROC_PID=$!
  PIDSTAT_THREAD_PID=
fi

{{
  echo "===== nvidia-smi initial ====="
  nvidia-smi || true
  echo
  echo "===== nvidia-smi pmon ====="
  nvidia-smi pmon -c "$DUR" || true
}} > "$OUT/gpu.txt" 2>&1 &
GPU_PID=$!

{{
  echo "===== ps top timeseries ====="
  END=$((SECONDS + DUR))
  while [ "$SECONDS" -lt "$END" ]; do
    echo
    echo "----- $(date +%H:%M:%S) -----"
    ps -eo pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -50
    sleep 5
  done
}} > "$OUT/top_processes_timeseries.txt" 2>&1 &
TOP_PID=$!

wait "$PIDSTAT_PROC_PID" || true
if [ -n "${{PIDSTAT_THREAD_PID:-}}" ]; then
  wait "$PIDSTAT_THREAD_PID" || true
fi
wait "$GPU_PID" || true
wait "$TOP_PID" || true

{{
  echo "===== top processes final ====="
  ps -eo pid,ppid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -80
}} > "$OUT/top_processes_final.txt" 2>&1

pgrep -f "launch_fusion_system|global_yolo_infer_server|global_yolo_merge_publisher|global_person_id_server|person_id_dataset_recorder|event_mask_evidence_service|global_bullet_reprocess_service|global_bullet_fusion_service|damage_attribution_service|foam_dart_speed_monitor_service|manual_hit_service|game_event_bridge_service|global_bev_fusion_renderer|event_overlay_overview_renderer|pyopengl_cuda_preview_renderer|mvs_cam_worker|mvs_trigger_coordinator|hik_rgb_yolo|hit_judge|multi_camera_launcher|metavision" | while read -r pid; do
  ps -p "$pid" -o pid,ppid,psr,pcpu,pmem,comm,args > "$OUT/threads/${{pid}}_process_final.txt" 2>&1 || true
  ps -L -p "$pid" -o pid,tid,psr,pcpu,pmem,comm --sort=-pcpu > "$OUT/threads/${{pid}}_threads_final.txt" 2>&1 || true
done

cp -a sync_ipc/global_bev_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_yolo_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_yolo_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_yolo_worker_*_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_yolo_worker_*_latest.json "$OUT/status/" 2>/dev/null || true
cp -a /dev/shm/global_bev_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_person_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_person_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/person_id_dataset_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/person_id_dataset_control.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/person_id_dataset_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_bullet_reprocess_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_bullet_reprocess_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/event_mask_evidence_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/event_mask_control.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/event_overlay_masked_*_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/event_overlay_masked_*_history.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/event_overlay_sparse_*_history.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/event_sync_map_ring_*.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/record_control.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/record_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/record_status_*.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_bullet_fusion_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/global_bullet_fusion_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/damage_attribution_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/damage_attribution_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/manual_hit_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/manual_hit_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/manual_hit_control.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/manual_hit_commands.jsonl "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/manual_hit_confirmed.jsonl "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/pseudo_hit_*.jsonl "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/pseudo_hit_all.jsonl "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/foam_dart_speed_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/foam_dart_speed_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/game_event_bridge_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/game_event_bridge_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/game_event_bridge_control.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/system_status_latest.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/operator_annotations.jsonl "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/replay_pack_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/yolo_chain_mode_status.json "$OUT/status/" 2>/dev/null || true
cp -a sync_ipc/trigger_status.json "$OUT/status/" 2>/dev/null || true
LATEST_STATUS_DIR="$(ls -td sync_ipc/status_logs/* 2>/dev/null | head -1 || true)"
if [ -n "$LATEST_STATUS_DIR" ]; then
  cp -a "$LATEST_STATUS_DIR/status_latest.json" "$OUT/status/status_latest.json" 2>/dev/null || true
  cp -a "$LATEST_STATUS_DIR/status_summary.csv" "$OUT/status/status_summary.csv" 2>/dev/null || true
  cp -a "$LATEST_STATUS_DIR/status_tabs.jsonl" "$OUT/status/status_tabs.jsonl" 2>/dev/null || true
fi
cp -a sync_ipc/launcher_logs/global_bev.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/global_yolo.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/global_yolo_*.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/global_yolo_merge.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/global_pid.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/person_id_dataset.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/global_bullet_reprocess.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/global_bullet_fusion.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/damage_attribution.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/foam_dart_speed.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/manual_hit.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/game_event_bridge.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/gl_cuda.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/event_overview.log "$OUT/logs/" 2>/dev/null || true
cp -a sync_ipc/launcher_logs/trigger.log "$OUT/logs/" 2>/dev/null || true

{{
  echo "===== latest status log dir ====="
  ls -td sync_ipc/status_logs/* 2>/dev/null | head -5 || true
  echo
  echo "===== global_bev_status ====="
  cat sync_ipc/global_bev_status.json 2>/dev/null || true
  echo
  echo "===== global_yolo_status ====="
  cat sync_ipc/global_yolo_status.json 2>/dev/null || true
  echo
  echo "===== global_person_status ====="
  cat sync_ipc/global_person_status.json 2>/dev/null || true
}} > "$OUT/runtime_status.txt" 2>&1

tar -czf "$OUT.tar.gz" "$OUT"
echo "[collect] done: $OUT.tar.gz"
ls -lh "$OUT.tar.gz"
"""
    return ['bash', '-lc', script]


def _flatten_launch_profile(data: Dict, prefix: str = '') -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not isinstance(data, dict):
        return out
    skip = {'description', 'notes', 'comment', 'comments', 'schema_version'}
    no_prefix = {'args', 'base', 'launcher', 'general'}
    for key, value in data.items():
        k = str(key).strip().replace('-', '_')
        if not k or k in skip:
            continue
        if isinstance(value, dict):
            if k in no_prefix:
                out.update(_flatten_launch_profile(value, prefix=''))
            else:
                next_prefix = f'{prefix}_{k}' if prefix else k
                out.update(_flatten_launch_profile(value, prefix=next_prefix))
            continue
        dest = f'{prefix}_{k}' if prefix else k
        out[dest] = value
    return out


def _apply_launch_profile_defaults(ap: argparse.ArgumentParser, profile_path: str, project_root_hint: str = '') -> Path:
    raw = str(profile_path or '').strip()
    if not raw:
        return Path()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        bases: List[Path] = []
        if project_root_hint:
            bases.append(Path(project_root_hint).expanduser())
        bases.extend([Path.cwd(), Path(__file__).resolve().parent])
        for base in bases:
            cand = (base / p).resolve()
            if cand.exists():
                p = cand
                break
        else:
            p = (bases[0] / p).resolve() if bases else p.resolve()
    data = load_json(p)
    defaults = _flatten_launch_profile(data)
    valid = {a.dest for a in ap._actions}
    invalid = sorted(k for k in defaults if k not in valid)
    if invalid:
        raise SystemExit(f"[launcher][ERROR] launch profile {p} contains unknown args: {', '.join(invalid[:20])}")
    ap.set_defaults(**defaults)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description='One-click launcher for event/MVS/hit/OpenGL fusion stack')
    ap.add_argument('--launch-profile', default='', help='JSON file whose values become launcher argument defaults; explicit CLI args still override it.')
    ap.add_argument('--project-root', default='', help='Project root. Default: directory containing this file')
    ap.add_argument('--cams', default='', help='Comma separated camera ids. Default: enabled event cameras, usually A,B,C,D,E')
    ap.add_argument('--config', default='ids_8cam_fusion_config.json', help='Single unified fusion JSON. Default: ids_8cam_fusion_config.json. Set empty string to use --event-config/--mvs-config directly.')
    ap.add_argument('--event-config', default='cameras_sync_live_D_v3_ts_align.json')
    ap.add_argument('--mvs-config', default='hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json')
    ap.add_argument('--preview-fps', type=float, default=60.0)
    ap.add_argument('--preview-backend', choices=['python-gl', 'pyopengl-cuda'], default='python-gl', help='Preview renderer backend. pyopengl-cuda uses raw Bayer shm + CUDA/PBO video pipeline.')
    ap.add_argument('--preview-video-fps', type=float, default=40.0, help='Video texture update FPS for pyopengl-cuda backend')
    ap.add_argument('--preview-render-fps', type=float, default=250.0, help='Renderer redraw FPS for pyopengl-cuda backend')
    ap.add_argument('--preview-trajectory-fps', type=float, default=250.0, help='Bullet trajectory update/redraw FPS for pyopengl-cuda backend')
    ap.add_argument('--preview-trajectory-life-ms', type=float, default=2500.0, help='Bullet trajectory visible TTL for pyopengl-cuda backend and hit_judge trajectory shm')
    ap.add_argument('--preview-trajectory-max-segments', type=int, default=8192, help='Preview trajectory shm ring size and renderer draw cap.')
    ap.add_argument('--preview-bullet-status-ttl-ms', type=float, default=0.0, help='Preview bullet diagnostics TTL; 0 lets the renderer choose its default.')
    ap.add_argument('--preview-vsync', type=int, default=0, help='Swap interval for pyopengl-cuda backend, 0 disables vsync')
    ap.add_argument('--preview-cuda-device', type=int, default=0)
    ap.add_argument('--preview-tile-width', type=int, default=640)
    ap.add_argument('--preview-tile-height', type=int, default=360)
    ap.add_argument('--preview-layout', default='grid2x4')
    ap.add_argument('--preview-window-width', type=int, default=2560)
    ap.add_argument('--preview-window-height', type=int, default=1440)
    ap.add_argument('--preview-raw-template', default='mvs_raw_latest_{camera}', help='Raw Bayer8 shm template for pyopengl-cuda backend')
    ap.add_argument('--preview-trajectory-shm', default='preview_trajectory')
    ap.add_argument('--preview-human-mask-render-mode', choices=['off', 'auto', 'polygon', 'bbox'], default='auto',
                    help='Preview-only human mask fill mode for pyopengl-cuda. auto uses polygons when present, otherwise bbox fallback.')
    ap.add_argument('--preview-human-mask-alpha', type=int, default=70, help='Preview-only human mask fill alpha 0-255.')
    ap.add_argument('--preview-human-mask-life-ms', type=float, default=5000.0, help='Preview-only human mask visible TTL.')
    ap.add_argument('--preview-human-label-mode', choices=['id_only', 'detail', 'off'], default='detail',
                    help='Preview-only human label mode for mask/bbox labels.')
    ap.add_argument('--global-bev-enable', action='store_true', help='Launch independent GPU global BEV fusion process.')
    ap.add_argument('--global-bev-sync-mode', choices=['nearest_trigger', 'nearest_frame'], default='nearest_trigger')
    ap.add_argument('--global-bev-max-mvs-sync-delta-ticks', type=int, default=2)
    ap.add_argument('--global-bev-sync-tolerance-ms', type=float, default=0.0)
    ap.add_argument('--global-bev-mvs-sync-outlier-policy', choices=['hide', 'warn'], default='hide')
    ap.add_argument('--global-bev-mvs-sync-tick-ms-est', type=float, default=25.0)
    ap.add_argument('--global-bev-presentation-delay-enable', action='store_true', default=False)
    ap.add_argument('--global-bev-no-presentation-delay-enable', dest='global_bev_presentation_delay_enable', action='store_false')
    ap.add_argument('--global-bev-presentation-delay-ms', type=float, default=0.0)
    ap.add_argument('--global-bev-alignment-mode', choices=['current', 'shadow', 'active'], default='current')
    ap.add_argument('--global-bev-alignment-delay-ms', type=float, default=820.0)
    ap.add_argument('--global-bev-alignment-shadow-interval-ms', type=float, default=250.0)
    ap.add_argument('--global-bev-alignment-hold-warn-ms', type=float, default=150.0)
    ap.add_argument('--global-bev-field-width-m', type=float, default=33.0)
    ap.add_argument('--global-bev-field-height-m', type=float, default=22.0)
    ap.add_argument('--global-bev-field-origin-x-m', type=float, default=0.0)
    ap.add_argument('--global-bev-field-origin-y-m', type=float, default=0.0)
    ap.add_argument('--global-bev-meters-per-pixel', type=float, default=0.02)
    ap.add_argument('--global-bev-display-fps', type=float, default=30.0)
    ap.add_argument('--global-bev-output-fps', type=float, default=20.0)
    ap.add_argument('--global-bev-perf-timing-window', type=int, default=180)
    ap.add_argument('--global-bev-pbo-count', type=int, default=3)
    ap.add_argument('--global-bev-fbo-stats-interval-ms', type=float, default=1000.0)
    ap.add_argument('--global-bev-fbo-stats-stride', type=int, default=8)
    ap.add_argument('--global-bev-coverage-interval-ms', type=float, default=2000.0)
    ap.add_argument('--global-bev-turf-stat-fps', type=float, default=5.0)
    ap.add_argument('--global-bev-person-overlay-build-fps', type=float, default=15.0)
    ap.add_argument('--global-bev-overlay-text-fps', type=float, default=5.0)
    ap.add_argument('--global-bev-ring-slots', type=int, default=8)
    ap.add_argument('--global-bev-window-width', type=int, default=1200)
    ap.add_argument('--global-bev-window-height', type=int, default=800)
    ap.add_argument('--global-bev-output-shm', default='global_bev_latest')
    ap.add_argument('--global-bev-sidecar-json', default='/dev/shm/global_bev_latest.json')
    ap.add_argument('--global-bev-status-json', default='sync_ipc/global_bev_status.json')
    ap.add_argument('--global-bev-control-json', default='sync_ipc/global_bev_control.json')
    ap.add_argument('--global-bev-runtime-state-json', default='sync_ipc/global_bev_runtime_state.json')
    ap.add_argument('--global-bev-runtime-state-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-runtime-state-enable', dest='global_bev_runtime_state_enable', action='store_false')
    ap.add_argument('--global-bev-runtime-state-event-layer-display-restore-enable', action='store_true', default=False)
    ap.add_argument('--global-bev-no-runtime-state-event-layer-display-restore-enable', dest='global_bev_runtime_state_event_layer_display_restore_enable', action='store_false')
    ap.add_argument('--global-bev-global-person-overlay-enable', action='store_true', help='Draw global_player_id labels from global_person_latest.json on the Global BEV window.')
    ap.add_argument('--global-bev-global-person-latest-json', default='sync_ipc/global_person_latest.json')
    ap.add_argument('--global-bev-global-person-overlay-coord-mode', choices=['field_xy', 'raw', 'neg_y', 'field_height_plus_y', 'invert_y', 'flip_x', 'flip_xy', 'swap_xy', 'swap_xy_neg_y'], default='field_xy')
    ap.add_argument('--global-bev-global-person-overlay-max-speed-mps', type=float, default=14.0)
    ap.add_argument('--global-bev-global-person-overlay-lost-max-age-ms', type=float, default=500.0)
    ap.add_argument('--global-bev-event-bullet-overlay-enable', action='store_true', help='Draw event-camera bullet points in Global BEV by inverse MVS calibration.')
    ap.add_argument('--global-bev-no-event-bullet-overlay-enable', dest='global_bev_event_bullet_overlay_enable', action='store_false')
    ap.add_argument('--global-bev-event-bullet-jsonl', default='sync_ipc/overlay_bullet_point_all.jsonl')
    ap.add_argument('--global-bev-event-bullet-overlay-life-ms', type=float, default=10000.0)
    ap.add_argument('--global-bev-event-bullet-overlay-max-records', type=int, default=4096)
    ap.add_argument('--global-bev-event-bullet-overlay-poll-ms', type=float, default=40.0)
    ap.add_argument('--global-bev-event-bullet-overlay-dilate-px', type=float, default=4.0)
    ap.add_argument('--global-bev-event-bullet-overlay-line-width', type=float, default=2.5)
    ap.add_argument('--global-bev-event-bullet-overlay-min-confidence', type=float, default=0.0)
    ap.add_argument('--global-bev-event-bullet-overlay-max-segment-gap-ms', type=float, default=80.0)
    ap.add_argument('--global-bev-event-bullet-overlay-label-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-bullet-overlay-label-enable', dest='global_bev_event_bullet_overlay_label_enable', action='store_false')
    ap.add_argument('--global-bev-event-bullet-overlay-transform-mode', choices=['field_xy', 'raw', 'normal', 'none', 'neg_y', 'field_height_plus_y', 'invert_y', 'flip_x', 'flip_y', 'flip_xy', 'swap_xy', 'swap_xy_neg_y'], default='normal')
    ap.add_argument('--global-bev-event-layer-enable', action='store_true', default=False, help='Overlay event-camera event layer in the Global BEV FBO with transparent black background.')
    ap.add_argument('--global-bev-no-event-layer-enable', dest='global_bev_event_layer_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-template', default='event_overlay_latest_{camera}')
    ap.add_argument('--global-bev-event-layer-sidecar-template', default='event_overlay_{camera}_latest.json')
    ap.add_argument('--global-bev-event-layer-backend', choices=['texture', 'sparse_gl'], default='texture')
    ap.add_argument('--global-bev-event-layer-sparse-sidecar-template', default='event_overlay_sparse_{camera}_history.json')
    ap.add_argument('--global-bev-event-layer-sparse-max-points-draw', type=int, default=20000)
    ap.add_argument('--global-bev-event-layer-sparse-candidate-scan-initial', type=int, default=4)
    ap.add_argument('--global-bev-event-layer-sparse-candidate-scan-escalated', type=int, default=32)
    ap.add_argument('--global-bev-event-layer-sparse-candidate-escalate-abs-ms', type=float, default=10.0)
    ap.add_argument('--global-bev-event-layer-sparse-candidate-escalate-ratio', type=float, default=0.60)
    ap.add_argument('--global-bev-event-layer-sparse-candidate-escalate-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-layer-sparse-candidate-escalate-enable', dest='global_bev_event_layer_sparse_candidate_escalate_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-sparse-shadow-enable', action='store_true', default=False)
    ap.add_argument('--global-bev-no-event-layer-sparse-shadow-enable', dest='global_bev_event_layer_sparse_shadow_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-sparse-shadow-interval-ms', type=float, default=250.0)
    ap.add_argument('--global-bev-event-layer-presentation-worker-enable', action='store_true', default=False)
    ap.add_argument('--global-bev-no-event-layer-presentation-worker-enable', dest='global_bev_event_layer_presentation_worker_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-presentation-worker-mode', choices=['off', 'shadow', 'active'], default='off')
    ap.add_argument('--global-bev-event-layer-presentation-worker-interval-ms', type=float, default=33.333)
    ap.add_argument('--global-bev-event-layer-presentation-bundle-max-age-ms', type=float, default=120.0)
    ap.add_argument('--global-bev-event-layer-presentation-target-tolerance-ticks', type=int, default=1)
    ap.add_argument('--global-bev-event-layer-presentation-bundle-ring-size', type=int, default=8)
    ap.add_argument('--global-bev-event-layer-presentation-prebuild-lookback-ticks', type=int, default=2)
    ap.add_argument('--global-bev-event-layer-presentation-prebuild-lookahead-ticks', type=int, default=2)
    ap.add_argument('--global-bev-event-layer-presentation-prebuild-max-per-cycle', type=int, default=1)
    ap.add_argument('--global-bev-event-layer-presentation-bundle-cache-reuse-ms', type=float, default=16.0)
    ap.add_argument('--global-bev-event-layer-homography-config', default='ids_8cam_fusion_config_627_runtime.json')
    ap.add_argument('--global-bev-event-layer-alpha', type=float, default=0.45)
    ap.add_argument('--global-bev-event-layer-gain', type=float, default=2.4)
    ap.add_argument('--global-bev-event-layer-min-value', type=float, default=0.03)
    ap.add_argument('--global-bev-event-layer-max-age-ms', type=float, default=150.0)
    ap.add_argument('--global-bev-event-layer-sync-mode', choices=['latest', 'latest_guarded', 'mvs_tick_guarded', 'drop_stale', 'fade'], default='latest_guarded')
    ap.add_argument('--global-bev-event-layer-max-sync-delta-ms', type=float, default=120.0)
    ap.add_argument('--global-bev-event-layer-max-sync-delta-ticks', type=int, default=2)
    ap.add_argument('--global-bev-event-layer-sync-normal-ms', type=float, default=50.0)
    ap.add_argument('--global-bev-event-layer-sync-warn-ms', type=float, default=80.0)
    ap.add_argument('--global-bev-event-layer-sync-hard-ms', type=float, default=80.0)
    ap.add_argument('--global-bev-event-layer-hold-last-good-on-hard-outlier', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-layer-hold-last-good-on-hard-outlier', dest='global_bev_event_layer_hold_last_good_on_hard_outlier', action='store_false')
    ap.add_argument('--global-bev-event-layer-hold-last-good-ms', type=float, default=250.0)
    ap.add_argument('--global-bev-event-layer-history-enable', action='store_true', default=False)
    ap.add_argument('--global-bev-no-event-layer-history-enable', dest='global_bev_event_layer_history_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-history-sidecar-template', default='')
    ap.add_argument('--global-bev-event-layer-sync-map-ring-template', default='event_sync_map_ring_{camera}.json')
    ap.add_argument('--global-bev-event-layer-fade-alpha-scale', type=float, default=0.25)
    ap.add_argument('--global-bev-event-layer-color-mode', choices=['camera', 'red', 'yellow'], default='camera')
    ap.add_argument('--global-bev-event-layer-dilate-px', type=float, default=16.0)
    ap.add_argument('--global-bev-event-layer-dilate-max-pixels', type=int, default=2500)
    ap.add_argument('--global-bev-event-layer-dilate-stabilize-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-layer-dilate-stabilize-enable', dest='global_bev_event_layer_dilate_stabilize_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-auto-degrade-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-layer-auto-degrade-enable', dest='global_bev_event_layer_auto_degrade_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-auto-degrade-fps-threshold', type=float, default=28.0)
    ap.add_argument('--global-bev-event-layer-auto-degrade-upload-p95-ms', type=float, default=8.0)
    ap.add_argument('--global-bev-event-layer-auto-degrade-min-dilate-px', type=float, default=4.0)
    ap.add_argument('--global-bev-event-layer-display-budget-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-layer-display-budget-enable', dest='global_bev_event_layer_display_budget_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-display-budget-ms', type=float, default=10.0)
    ap.add_argument('--global-bev-event-layer-display-hold-ms', type=float, default=280.0)
    ap.add_argument('--global-bev-event-layer-accumulate-enable', action='store_true', default=True)
    ap.add_argument('--global-bev-no-event-layer-accumulate-enable', dest='global_bev_event_layer_accumulate_enable', action='store_false')
    ap.add_argument('--global-bev-event-layer-accumulate-window-ms', type=float, default=40.0)
    ap.add_argument('--global-bev-event-layer-accumulate-max-frames', type=int, default=4)
    ap.add_argument('--global-bev-event-layer-texture-scale', type=float, default=1.0)
    ap.add_argument('--global-bev-event-layer-history-poll-interval-ms', type=float, default=0.0)
    ap.add_argument('--global-bev-status-debug-interval-ms', type=float, default=0.0)
    ap.add_argument('--global-bev-calib-template', default='calib_true/field_calib_camera{camera}.json')
    ap.add_argument('--global-bev-cams', default='', help='Optional camera subset for GLOBAL_BEV only, e.g. A or A,C,E,G. Default uses --cams.')
    ap.add_argument('--global-bev-cpus', default='', help='Optional CPU set for independent GLOBAL_BEV process, e.g. 18-19.')
    ap.add_argument('--global-bev-debug-gray', action='store_true', help='Render BEV raw grayscale instead of Bayer color for geometry/debug runs.')
    ap.add_argument('--global-bev-debug-mode', choices=['warp_color', 'warp_gray', 'coverage', 'raw_mosaic', 'weight_heatmap', 'cam_index', 'uv_heatmap'], default='warp_color',
                    help='GLOBAL_BEV debug/render mode. coverage/cam_index/weight_heatmap/uv_heatmap check geometry, raw_mosaic checks texture upload, warp_gray/warp_color render BEV.')
    ap.add_argument('--global-bev-texture-y-flip', action='store_true', help='Flip raw texture Y lookup in GLOBAL_BEV shader.')
    ap.add_argument('--global-bev-no-texture-y-flip', dest='global_bev_texture_y_flip', action='store_false', help='Disable GLOBAL_BEV texture Y flip for orientation experiments.')
    ap.add_argument('--global-bev-texture-format', choices=['r8', 'luminance'], default='r8', help='Raw texture upload format for GLOBAL_BEV.')
    ap.add_argument('--global-bev-gain', type=float, default=1.0, help='Tone gain for GLOBAL_BEV warp_color/warp_gray modes.')
    ap.add_argument('--global-bev-black-level', type=float, default=0.0, help='Tone black level subtracted before GLOBAL_BEV gain.')
    ap.add_argument('--global-bev-gamma', type=float, default=1.0, help='Tone gamma for GLOBAL_BEV warp_color/warp_gray modes.')
    ap.add_argument('--global-bev-color-gain-r', type=float, default=1.0, help='GLOBAL_BEV post-demosaic red channel gain.')
    ap.add_argument('--global-bev-color-gain-g', type=float, default=1.0, help='GLOBAL_BEV post-demosaic green channel gain.')
    ap.add_argument('--global-bev-color-gain-b', type=float, default=1.0, help='GLOBAL_BEV post-demosaic blue channel gain.')
    ap.add_argument('--global-bev-turf-match-enable', action='store_true', help='Enable grass/turf color matching for GLOBAL_BEV live display.')
    ap.add_argument('--global-bev-turf-auto-match', action='store_true', help='Estimate per-camera turf color gains from raw frames.')
    ap.add_argument('--global-bev-turf-match-strength', type=float, default=0.28, help='Strength of green turf hue normalization in GLOBAL_BEV.')
    ap.add_argument('--global-bev-turf-target-rgb', default='', help='Manual target raw RGB for turf color matching, e.g. 0.18,0.42,0.15.')
    ap.add_argument('--global-bev-frame-hold-ms', type=float, default=250.0, help='Keep last successfully uploaded camera texture for this long to avoid single-camera flicker.')
    ap.add_argument('--global-bev-unstable-copy-retries', type=int, default=3, help='Retries for stable raw ring slot copy before skipping a camera texture update.')
    ap.add_argument('--global-bev-mvs-upload-budget-enable', action='store_true', default=True, help='Keep previous MVS texture when per-frame BEV upload budget is exhausted.')
    ap.add_argument('--global-bev-no-mvs-upload-budget-enable', dest='global_bev_mvs_upload_budget_enable', action='store_false')
    ap.add_argument('--global-bev-mvs-upload-budget-ms', type=float, default=10.0, help='Global BEV GUI-thread MVS texture upload budget per paint tick.')
    ap.add_argument('--global-bev-grid-enable', action='store_true', help='Overlay 1m/5m field grid in GLOBAL_BEV.')
    ap.add_argument('--global-bev-field-transform-mode',
                    choices=['normal', 'flip_x', 'flip_y', 'flip_xy', 'swap_xy', 'swap_xy_flip_x', 'swap_xy_flip_y', 'swap_xy_flip_xy'],
                    default='normal',
                    help='Runtime GLOBAL_BEV display-to-field transform. Can also be hot-switched from sync_ipc/global_bev_control.json.')
    ap.add_argument('--global-bev-field-flip-x', action='store_true', help='Flip GLOBAL_BEV display X before sampling calibrated field coordinates.')
    ap.add_argument('--global-bev-field-flip-y', action='store_true', help='Flip GLOBAL_BEV display Y before sampling calibrated field coordinates.')
    ap.add_argument('--global-person-id-enable', action='store_true', help='Launch independent global physical-space person ID server.')
    ap.add_argument('--global-person-id-cpus', default='', help='Optional CPU set for global person ID process, e.g. 28-29.')
    ap.add_argument('--global-person-id-human-jsonl-template', default='{root}/human_result_{camera}.jsonl')
    ap.add_argument('--global-person-id-output-jsonl', default='global_person_tracks.jsonl')
    ap.add_argument('--global-person-id-latest-json', default='global_person_latest.json')
    ap.add_argument('--global-person-id-status-json', default='global_person_status.json')
    ap.add_argument('--global-person-id-control-json', default='global_person_id_control.json')
    ap.add_argument('--global-person-id-control-poll-ms', type=float, default=200.0)
    ap.add_argument('--global-person-id-poll-ms', type=float, default=10.0)
    ap.add_argument('--global-person-id-sync-delay', type=int, default=1)
    ap.add_argument('--global-person-id-coord-transform-mode', choices=['raw', 'neg_y', 'field_height_plus_y', 'invert_y', 'flip_x', 'flip_xy', 'swap_xy', 'swap_xy_neg_y'], default='raw')
    ap.add_argument('--global-person-id-field-margin-m', type=float, default=2.0)
    ap.add_argument('--global-person-id-drop-out-of-field', action='store_true')
    ap.add_argument('--global-person-id-max-match-speed-mps', type=float, default=12.0)
    ap.add_argument('--global-person-id-max-update-speed-mps', type=float, default=10.0)
    ap.add_argument('--global-person-id-publish-lost-max-age-ms', type=float, default=500.0)
    ap.add_argument('--global-person-id-publish-min-hits', type=int, default=2)
    ap.add_argument('--global-person-id-cluster-gate-m', type=float, default=0.65)
    ap.add_argument('--global-person-id-same-cam-cluster-gate-m', type=float, default=0.22)
    ap.add_argument('--global-person-id-cross-camera-cluster-gate-m', type=float, default=0.75)
    ap.add_argument('--global-person-id-handoff-aware-enable', action='store_true', default=True)
    ap.add_argument('--global-person-id-no-handoff-aware-enable', dest='global_person_id_handoff_aware_enable', action='store_false')
    ap.add_argument('--global-person-id-handoff-match-gate-m', type=float, default=1.35)
    ap.add_argument('--global-person-id-handoff-birth-near-track-suppress-m', type=float, default=1.25)
    ap.add_argument('--global-person-id-handoff-duplicate-suppress-dist-m', type=float, default=1.35)
    ap.add_argument('--global-person-id-handoff-strict-count-lock-enable', action='store_true', default=True)
    ap.add_argument('--global-person-id-no-handoff-strict-count-lock-enable', dest='global_person_id_handoff_strict_count_lock_enable', action='store_false')
    ap.add_argument('--global-person-id-handoff-strict-birth-block-m', type=float, default=3.0)
    ap.add_argument('--global-person-id-handoff-strict-publish-hold-ms', type=float, default=1800.0)
    ap.add_argument('--global-person-id-count-gate-enable', action='store_true', default=True)
    ap.add_argument('--global-person-id-no-count-gate-enable', dest='global_person_id_count_gate_enable', action='store_false')
    ap.add_argument('--global-person-id-count-gate-x-min-m', type=float, default=0.0)
    ap.add_argument('--global-person-id-count-gate-x-max-m', type=float, default=30.0)
    ap.add_argument('--global-person-id-count-gate-y-min-m', type=float, default=0.0)
    ap.add_argument('--global-person-id-count-gate-y-max-m', type=float, default=20.0)
    ap.add_argument('--global-person-id-count-gate-short-edge-band-m', type=float, default=3.0)
    ap.add_argument('--global-person-id-count-gate-short-edge-y-pad-m', type=float, default=2.0)
    ap.add_argument('--global-person-id-count-gate-startup-bootstrap-ms', type=float, default=5000.0)
    ap.add_argument('--global-person-id-count-gate-bootstrap-allow-anywhere-until-track-count', type=int, default=8)
    ap.add_argument('--global-person-id-count-gate-non-short-edge-lost-hold-ms', type=float, default=5000.0)
    ap.add_argument('--global-person-id-count-gate-non-short-edge-retire-hard-cap-ms', type=float, default=15000.0)
    ap.add_argument('--global-person-id-duplicate-suppress-enable', action='store_true', default=True)
    ap.add_argument('--global-person-id-no-duplicate-suppress-enable', dest='global_person_id_duplicate_suppress_enable', action='store_false')
    ap.add_argument('--global-person-id-duplicate-suppress-dist-m', type=float, default=0.9)
    ap.add_argument('--global-person-id-match-gate-m', type=float, default=1.25)
    ap.add_argument('--global-person-id-lost-match-gate-m', type=float, default=1.8)
    ap.add_argument('--global-person-id-match-score-min', type=float, default=0.32)
    ap.add_argument('--global-person-id-birth-quarantine-enable', action='store_true', default=True)
    ap.add_argument('--global-person-id-no-birth-quarantine-enable', dest='global_person_id_birth_quarantine_enable', action='store_false')
    ap.add_argument('--global-person-id-birth-confirm-hits', type=int, default=2)
    ap.add_argument('--global-person-id-birth-match-gate-m', type=float, default=0.85)
    ap.add_argument('--global-person-id-birth-near-track-suppress-m', type=float, default=0.65)
    ap.add_argument('--global-person-id-birth-candidate-ttl-ms', type=float, default=900.0)
    ap.add_argument('--global-person-id-reid-memory-ms', type=float, default=3000.0)
    ap.add_argument('--global-person-id-retire-after-ms', type=float, default=12000.0)
    ap.add_argument('--global-person-id-filter-backend', choices=['alpha_beta', 'kalman_cv'], default='alpha_beta')
    ap.add_argument('--global-person-id-position-alpha', type=float, default=0.65)
    ap.add_argument('--global-person-id-velocity-alpha', type=float, default=0.25)
    ap.add_argument('--global-person-id-kalman-use-for-association', action='store_true', default=False)
    ap.add_argument('--global-person-id-no-kalman-use-for-association', dest='global_person_id_kalman_use_for_association', action='store_false')
    ap.add_argument('--global-person-id-kalman-process-noise-m', type=float, default=0.35)
    ap.add_argument('--global-person-id-kalman-measurement-noise-m', type=float, default=0.45)
    ap.add_argument('--global-person-id-kalman-initial-pos-var', type=float, default=0.6)
    ap.add_argument('--global-person-id-kalman-initial-vel-var', type=float, default=4.0)
    ap.add_argument('--global-person-id-kalman-max-innovation-m', type=float, default=2.5)
    ap.add_argument('--global-person-id-kalman-reject-innovation-enable', action='store_true', default=False)
    ap.add_argument('--global-person-id-no-kalman-reject-innovation-enable', dest='global_person_id_kalman_reject_innovation_enable', action='store_false')
    ap.add_argument('--global-person-id-shadow-enable', action='store_true', default=False, help='Launch a non-authoritative Global Person ID shadow stream for hot A/B tests.')
    ap.add_argument('--disable-global-person-id-shadow', dest='global_person_id_shadow_enable', action='store_false')
    ap.add_argument('--global-person-id-shadow-cpus', default='', help='Optional CPU set for shadow Global Person ID process.')
    ap.add_argument('--global-person-id-shadow-output-jsonl', default='global_person_shadow_tracks.jsonl')
    ap.add_argument('--global-person-id-shadow-latest-json', default='global_person_shadow_latest.json')
    ap.add_argument('--global-person-id-shadow-status-json', default='global_person_shadow_status.json')
    ap.add_argument('--global-person-id-shadow-control-json', default='global_person_shadow_control.json')
    ap.add_argument('--global-person-id-shadow-filter-backend', choices=['alpha_beta', 'kalman_cv'], default='kalman_cv')
    ap.add_argument('--person-id-dataset-recorder-enable', action='store_true', default=False, help='Launch idle YOLO-after-recognition Person ID dataset recorder sidecar.')
    ap.add_argument('--disable-person-id-dataset-recorder', dest='person_id_dataset_recorder_enable', action='store_false')
    ap.add_argument('--person-id-dataset-recorder-cpus', default='', help='Optional CPU set for the Person ID dataset recorder.')
    ap.add_argument('--person-id-dataset-output-root', default='recordings/person_id_datasets')
    ap.add_argument('--person-id-dataset-evidence-output-root', default='recordings/evidence_replay_datasets')
    ap.add_argument('--person-id-dataset-control-json', default='person_id_dataset_control.json')
    ap.add_argument('--person-id-dataset-status-json', default='person_id_dataset_status.json')
    ap.add_argument('--person-id-dataset-latest-json', default='person_id_dataset_latest.json')
    ap.add_argument('--person-id-dataset-poll-ms', type=float, default=100.0)
    ap.add_argument('--person-id-dataset-status-interval-ms', type=float, default=1000.0)
    ap.add_argument('--person-id-dataset-sample-status-interval-ms', type=float, default=1000.0)
    ap.add_argument('--person-id-dataset-zero-person-keep-ms', type=float, default=0.0)
    ap.add_argument('--global-bullet-reprocess-enable', action='store_true', default=False, help='Launch Global Bullet Reprocess sidecar in diagnostic/shadow mode.')
    ap.add_argument('--disable-global-bullet-reprocess', dest='global_bullet_reprocess_enable', action='store_false')
    ap.add_argument('--global-bullet-reprocess-mode', choices=['capture_only', 'shadow', 'idle'], default='shadow')
    ap.add_argument('--global-bullet-reprocess-cpus', default='', help='Optional CPU set for Global Bullet Reprocess sidecar.')
    ap.add_argument('--global-bullet-reprocess-output-dir', default='sync_ipc/global_bullet_reprocess')
    ap.add_argument('--global-bullet-reprocess-interval-sec', type=float, default=1.0)
    ap.add_argument('--global-bullet-reprocess-duration-sec', type=float, default=0.0, help='0 or negative means run until launcher stops.')
    ap.add_argument('--global-bullet-reprocess-max-scan-lines', type=int, default=100000)
    ap.add_argument('--global-bullet-reprocess-calib-manifest', default='')
    ap.add_argument('--global-bullet-reprocess-allow-empty', action='store_true', default=True)
    ap.add_argument('--global-bullet-reprocess-no-allow-empty', dest='global_bullet_reprocess_allow_empty', action='store_false')
    ap.add_argument('--global-bullet-fusion-enable', action='store_true', default=False, help='A-test sidecar: fuse event bullet points into global field tracks.')
    ap.add_argument('--disable-global-bullet-fusion', dest='global_bullet_fusion_enable', action='store_false')
    ap.add_argument('--global-bullet-fusion-mode', choices=['disabled', 'shadow_no_publish', 'shadow', 'publish', 'enforce'], default='disabled')
    ap.add_argument('--global-bullet-fusion-cpus', default='', help='Optional CPU set for global bullet fusion sidecar.')
    ap.add_argument('--global-bullet-fusion-interval-sec', type=float, default=0.04)
    ap.add_argument('--global-bullet-fusion-input-jsonl', default='overlay_bullet_point_all.jsonl')
    ap.add_argument('--global-bullet-fusion-output-jsonl', default='global_bullet_tracks.jsonl')
    ap.add_argument('--global-bullet-fusion-latest-json', default='global_bullet_fusion_latest.json')
    ap.add_argument('--global-bullet-fusion-status-json', default='global_bullet_fusion_status.json')
    ap.add_argument('--global-bullet-fusion-calib-template', default='calib_true/field_calib_camera{camera}.json')
    ap.add_argument('--global-bullet-fusion-transform-mode', default='flip_y')
    ap.add_argument('--global-bullet-fusion-min-points', type=int, default=3)
    ap.add_argument('--global-bullet-fusion-track-ttl-ms', type=float, default=1500.0)
    ap.add_argument('--damage-attribution-enable', action='store_true', default=False, help='A-test sidecar: score likely shooter/source from global bullet tracks and Global Person ID.')
    ap.add_argument('--disable-damage-attribution', dest='damage_attribution_enable', action='store_false')
    ap.add_argument('--damage-attribution-mode', choices=['disabled', 'shadow_no_publish', 'shadow', 'publish', 'enforce'], default='disabled')
    ap.add_argument('--damage-attribution-cpus', default='', help='Optional CPU set for damage attribution sidecar.')
    ap.add_argument('--damage-attribution-interval-sec', type=float, default=0.08)
    ap.add_argument('--damage-attribution-score-threshold', type=float, default=0.45)
    ap.add_argument('--damage-attribution-hit-jsonl', default='hit_candidate_all.jsonl')
    ap.add_argument('--damage-attribution-events-jsonl', default='damage_attribution_events.jsonl')
    ap.add_argument('--damage-attribution-latest-json', default='damage_attribution_latest.json')
    ap.add_argument('--damage-attribution-status-json', default='damage_attribution_status.json')
    ap.add_argument('--damage-attribution-source', choices=['global_bullet', 'direct_hit_global_id'], default='global_bullet')
    ap.add_argument('--damage-attribution-target-id-source', choices=['global_runtime', 'global_bev_display'], default='global_runtime')
    ap.add_argument('--damage-attribution-global-bev-status-json', default='global_bev_status.json')
    ap.add_argument('--damage-attribution-global-bev-display-max-age-ms', type=float, default=1200.0)
    ap.add_argument('--damage-attribution-default-source-global-id', type=int, default=99)
    ap.add_argument('--damage-attribution-person-max-age-ms', type=float, default=800.0)
    ap.add_argument('--damage-attribution-person-history-max-snapshots', type=int, default=2048)
    ap.add_argument('--damage-attribution-identity-evidence-max-sync-delta', type=int, default=8)
    ap.add_argument('--damage-attribution-identity-evidence-history-max-age-ms', type=float, default=5000.0)
    ap.add_argument('--damage-attribution-identity-spatial-fallback-enable', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--damage-attribution-identity-spatial-fallback-require-camera', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--damage-attribution-identity-spatial-fallback-max-dist-m', type=float, default=1.2)
    ap.add_argument('--damage-attribution-identity-spatial-fallback-min-gap-m', type=float, default=0.25)
    ap.add_argument('--damage-attribution-identity-spatial-fallback-max-sync-delta', type=int, default=16)
    ap.add_argument('--damage-attribution-display-history-max-snapshots', type=int, default=2048)
    ap.add_argument('--damage-attribution-display-history-max-age-ms', type=float, default=5000.0)
    ap.add_argument('--damage-attribution-allow-stable-id-global-fallback', action='store_true', default=False)
    ap.add_argument('--damage-attribution-reserved-virtual-global-id', type=int, default=-1)
    ap.add_argument('--manual-hit-service-enable', action='store_true', default=False, help='A-test sidecar: convert main-preview operator clicks into publishable hit events.')
    ap.add_argument('--disable-manual-hit-service', dest='manual_hit_service_enable', action='store_false')
    ap.add_argument('--manual-hit-mode', choices=['disabled', 'shadow_no_publish', 'shadow', 'publish', 'enforce'], default='disabled')
    ap.add_argument('--manual-hit-cpus', default='', help='Optional CPU set for manual hit sidecar.')
    ap.add_argument('--manual-hit-interval-sec', type=float, default=0.04)
    ap.add_argument('--manual-hit-status-interval-sec', type=float, default=0.5)
    ap.add_argument('--manual-hit-command-jsonl', default='manual_hit_commands.jsonl')
    ap.add_argument('--manual-hit-confirmed-jsonl', default='manual_hit_confirmed.jsonl')
    ap.add_argument('--manual-hit-events-jsonl', default='damage_attribution_events.jsonl')
    ap.add_argument('--manual-hit-latest-json', default='manual_hit_latest.json')
    ap.add_argument('--manual-hit-status-json', default='manual_hit_status.json')
    ap.add_argument('--manual-hit-control-json', default='manual_hit_control.json')
    ap.add_argument('--manual-hit-source-global-id', type=int, default=99)
    ap.add_argument('--manual-hit-dedup-ms', type=float, default=800.0)
    ap.add_argument('--manual-hit-target-max-age-ms', type=float, default=1200.0)
    ap.add_argument('--manual-hit-target-match-max-dist-m', type=float, default=1.8)
    ap.add_argument('--preview-manual-hit-click-enable', action='store_true', default=False, help='Enable main-preview click-to-manual-hit on GL_CUDA startup.')
    ap.add_argument('--preview-manual-hit-click-disable', dest='preview_manual_hit_click_enable', action='store_false')
    ap.add_argument('--foam-dart-speed-enable', action='store_true', default=False, help='A-test sidecar: estimate foam dart speed and produce overspeed diagnostics.')
    ap.add_argument('--disable-foam-dart-speed', dest='foam_dart_speed_enable', action='store_false')
    ap.add_argument('--foam-dart-speed-mode', choices=['disabled', 'shadow_no_publish', 'shadow', 'publish', 'enforce'], default='disabled')
    ap.add_argument('--foam-dart-speed-cpus', default='', help='Optional CPU set for foam dart speed monitor sidecar.')
    ap.add_argument('--foam-dart-speed-interval-sec', type=float, default=0.1)
    ap.add_argument('--foam-dart-speed-limit-mps', type=float, default=45.0)
    ap.add_argument('--foam-dart-speed-warn-mps', type=float, default=35.0)
    ap.add_argument('--game-event-bridge-enable', action='store_true', default=False, help='A-test sidecar: bridge attribution/speed diagnostics to game-event schema.')
    ap.add_argument('--disable-game-event-bridge', dest='game_event_bridge_enable', action='store_false')
    ap.add_argument('--game-event-bridge-mode', choices=['disabled', 'shadow_no_publish', 'shadow', 'publish', 'enforce'], default='disabled')
    ap.add_argument('--game-event-bridge-cpus', default='', help='Optional CPU set for game event bridge sidecar.')
    ap.add_argument('--game-event-bridge-interval-sec', type=float, default=0.1)
    ap.add_argument('--game-event-bridge-id-source', choices=['global_runtime', 'global_bev_display'], default='global_runtime')
    ap.add_argument('--game-event-bridge-global-bev-status-json', default='global_bev_status.json')
    ap.add_argument('--game-event-bridge-global-bev-display-max-age-ms', type=float, default=1200.0)
    ap.add_argument('--game-event-bridge-windows-host', default='192.168.31.76')
    ap.add_argument('--game-event-bridge-windows-port', type=int, default=7002)
    ap.add_argument('--game-event-bridge-presence-hz', type=float, default=5.0)
    ap.add_argument('--game-event-bridge-presence-max-age-ms', type=float, default=800.0)
    ap.add_argument('--game-event-bridge-send-presence-in-shadow', action='store_true', default=False)
    ap.add_argument('--game-event-bridge-reconnect-sec', type=float, default=3.0)
    ap.add_argument('--game-event-bridge-events-jsonl', default='damage_attribution_events.jsonl')
    ap.add_argument('--game-event-bridge-control-json', default='game_event_bridge_control.json')
    ap.add_argument('--game-event-bridge-hit-source-mode', choices=['all', 'manual_only', 'auto_only'], default='all')
    ap.add_argument('--game-event-bridge-latest-json', default='game_event_bridge_latest.json')
    ap.add_argument('--game-event-bridge-status-json', default='game_event_bridge_status.json')
    ap.add_argument('--game-event-bridge-sent-jsonl', default='game_event_bridge_sent.jsonl')
    ap.add_argument('--game-event-bridge-force-hit-source-global-id', type=int, default=-1)
    ap.add_argument('--game-event-bridge-virtual-presence-global-id', type=int, default=-1)
    ap.add_argument('--game-event-bridge-virtual-presence-player-id', default='')
    ap.add_argument('--auto-cpu-hotspots-enable', action='store_true', help='After all launcher children are started, collect CPU/GPU hotspot diagnostics automatically.')
    ap.add_argument('--auto-cpu-hotspots-delay-sec', type=float, default=5.0, help='Delay before automatic CPU hotspot collection starts.')
    ap.add_argument('--auto-cpu-hotspots-duration-sec', type=int, default=60, help='Automatic CPU hotspot collection duration in seconds.')
    ap.add_argument('--auto-cpu-hotspots-prefix', default='cpu_hotspots', help='Output directory/tar prefix for automatic CPU hotspot collection.')
    ap.add_argument('--event-overlay-preview-enable', action='store_true', help='Forward to pyopengl-cuda renderer: show event overlay in independent overview window')
    ap.add_argument('--event-overlay-main-inline-enable', action='store_true', help='Debug fallback: also overlay event visualization on RGB preview tiles')
    ap.add_argument('--event-overlay-window-disable', action='store_true')
    ap.add_argument('--event-overlay-window-width', type=int, default=1280)
    ap.add_argument('--event-overlay-window-height', type=int, default=720)
    ap.add_argument('--event-overlay-window-x', type=int, default=-100000)
    ap.add_argument('--event-overlay-window-y', type=int, default=-100000)
    ap.add_argument('--event-overlay-window-fps', type=float, default=50.0)
    ap.add_argument('--event-overlay-window-font-px', type=int, default=12)
    ap.add_argument('--event-overlay-window-layout', default='grid2x4')
    ap.add_argument('--event-overlay-preview-alpha', type=float, default=0.55)
    ap.add_argument('--event-overlay-preview-max-age-ms', type=float, default=100.0)
    ap.add_argument('--event-overlay-sync-mode', choices=['latest', 'nearest_mvs', 'nearest_human'], default='nearest_mvs')
    ap.add_argument('--event-overlay-max-sync-delta-ms', type=float, default=50.0)
    ap.add_argument('--event-overlay-hide-stale', action='store_true')
    ap.add_argument('--event-overlay-show-stats', action='store_true', default=True)
    ap.add_argument('--event-overlay-preview-template', default='event_overlay_latest_{camera},event_overlay_masked_latest_{camera}')
    ap.add_argument('--event-overlay-sidecar-template', default='event_overlay_{camera}_latest.json,event_overlay_masked_{camera}_latest.json')
    ap.add_argument('--event-overlay-preview-gain', type=float, default=3.0)
    ap.add_argument('--event-overlay-preview-gamma', type=float, default=0.65)
    ap.add_argument('--event-overlay-preview-min-alpha', type=float, default=0.35)
    ap.add_argument('--event-overlay-preview-dilate-px', type=int, default=4)
    ap.add_argument('--event-overlay-preview-dilate-max-pixels', type=int, default=5000)
    ap.add_argument('--event-overlay-preview-dilate-backend', choices=['gl_shader', 'cpu_sparse', 'off', 'auto'], default='gl_shader')
    ap.add_argument('--preview-allow-cpu-upload-fallback', action='store_true', help='Allow pyopengl-cuda renderer to fall back to CPU texture upload if CUDA-GL interop fails')
    ap.add_argument('--preview-allow-non-nvidia-gl', action='store_true', help='Debug only: allow GL context not reported as NVIDIA')
    ap.add_argument('--no-force-raise-windows', action='store_true', help='Do not force-raise preview/status/overlay windows after creation')
    ap.add_argument('--status-log-enable', action='store_true', help='Forward to pyopengl-cuda renderer: save status diagnostics to JSONL/CSV')
    ap.add_argument('--status-log-dir', default='', help='Forward to pyopengl-cuda renderer: status log directory; default <sync_root>/status_logs')
    ap.add_argument('--status-log-interval-ms', type=float, default=1000.0, help='Forward to pyopengl-cuda renderer: status log interval in ms')
    ap.add_argument('--status-log-tabs-jsonl', action='store_true', help='Forward to pyopengl-cuda renderer: also save status tab rows to JSONL')
    ap.add_argument('--status-log-latest-disable', action='store_true', help='Forward to pyopengl-cuda renderer: disable status_latest.json')
    ap.add_argument('--status-window-disable', action='store_true', help='Forward to pyopengl-cuda renderer: disable diagnostics status window')
    ap.add_argument('--status-window-poll-ms', type=float, default=500.0, help='Forward to pyopengl-cuda renderer: diagnostics status window refresh interval')
    ap.add_argument('--hit-port', type=int, default=5003)
    ap.add_argument('--hit-control-json', default='sync_ipc/hit_judge_control.json')
    ap.add_argument('--overlay-ws-enable', action='store_true', help='Enable UE5 overlay WebSocket broadcast from hit_judge_server')
    ap.add_argument('--overlay-ws-host', default='0.0.0.0')
    ap.add_argument('--overlay-ws-port', type=int, default=8765)
    ap.add_argument('--skip-event', action='store_true')
    ap.add_argument('--skip-mvs', action='store_true')
    ap.add_argument('--skip-hit', action='store_true')
    ap.add_argument('--skip-gl', action='store_true')
    ap.add_argument('--mvs-split-workers', action='store_true', help='Run one MVS camera worker per cam plus a trigger coordinator; the MVS/YOLO process reads frames from shm.')
    ap.add_argument('--mvs-trigger-fps', type=float, default=0.0, help='Trigger coordinator FPS. Default: sync.mvs_soft_trigger_fps or 1000/mvs_soft_trigger_period_ms.')
    ap.add_argument('--mvs-trigger-lead-ms', type=float, default=5.0, help='Coordinator sends future trigger ticks this many ms before fire time.')
    ap.add_argument('--mvs-trigger-start-delay-ms', type=float, default=40000.0, help='Delay before formal sync_start and first coordinated trigger sequence.')
    ap.add_argument('--mvs-trigger-sync-start-settle-ms', type=float, default=1000.0, help='Delay after writing sync_start.flag before coordinator sends tick_id=0.')
    ap.add_argument('--mvs-worker-ready-delay', type=float, default=2.0, help='Delay after starting split MVS workers before starting trigger coordinator.')
    ap.add_argument('--event-ready-settle-delay', type=float, default=1.0, help='Extra delay after all fresh event_ready flags before starting MVS/trigger.')
    ap.add_argument('--event-worker-cpus', default='', help='Optional comma-separated CPU cores for event camera workers, e.g. 30,31,32,33,34,35,36,37.')
    ap.add_argument('--event-empty-slice-fastpath-enable', action='store_true', default=True, help='Forward to event workers: skip heavy processing on empty slices (default enabled).')
    ap.add_argument('--disable-event-empty-slice-fastpath', dest='event_empty_slice_fastpath_enable', action='store_false')
    ap.add_argument('--event-small-slice-event-thresh', type=int, default=0)
    ap.add_argument('--event-human-mask-cache-enable', action='store_true', default=True, help='Forward to event workers: cache/trottle human mask JSONL polling (default enabled).')
    ap.add_argument('--disable-event-human-mask-cache', dest='event_human_mask_cache_enable', action='store_false')
    ap.add_argument('--event-human-mask-cache-max-fps', type=float, default=40.0)
    ap.add_argument('--event-human-mask-cache-stale-ms', type=float, default=150.0)
    ap.add_argument('--event-human-mask-buffer-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-human-mask-buffer', dest='event_human_mask_buffer_enable', action='store_false')
    ap.add_argument('--event-human-mask-buffer-ms', type=float, default=90.0)
    ap.add_argument('--event-human-mask-buffer-timeout-ms', type=float, default=150.0)
    ap.add_argument('--event-human-mask-buffer-release-policy', choices=['raw_at_budget', 'raw_at_timeout', 'strict_same_sync_drop'], default='raw_at_budget')
    ap.add_argument('--event-human-mask-buffer-max-events', type=int, default=300000)
    ap.add_argument('--event-human-mask-buffer-max-buckets', type=int, default=8)
    ap.add_argument('--event-human-mask-buffer-empty-release-fps', type=float, default=0.0)
    ap.add_argument('--event-human-mask-filter-max-fps', type=float, default=0.0)
    ap.add_argument('--event-human-mask-stats-jsonl', default='', help='Forward to event workers: optional human-mask diagnostics JSONL. Empty disables it.')
    ap.add_argument('--event-human-mask-stats-max-fps', type=float, default=2.0, help='Forward to event workers: max human-mask diagnostics JSONL write FPS when enabled.')
    ap.add_argument('--event-human-mask-stats-flush-ms', type=float, default=1000.0, help='Forward to event workers: human-mask diagnostics JSONL flush interval.')
    ap.add_argument('--event-sync-source', default='mvs_soft_trigger', choices=['event_local', 'mvs_soft_trigger', 'mvs_timeline', 'mvs_soft_trigger_timeline'])
    ap.add_argument('--event-sync-wall-guard-ms', type=float, default=20.0)
    ap.add_argument('--event-sync-offset-jump-warn-frames', type=int, default=2)
    ap.add_argument('--event-sync-local-fallback-enable', action='store_true', default=True)
    ap.add_argument('--disable-event-sync-local-fallback', dest='event_sync_local_fallback_enable', action='store_false')
    ap.add_argument('--tsf1-pub-fps', type=float, default=8.0)
    ap.add_argument('--tsf1-pub-accumulation-time-us', type=int, default=12000)
    ap.add_argument('--event-overlay-pub-max-fps', type=float, default=170.0)
    ap.add_argument('--event-overlay-pub-channels', type=int, choices=[1, 3], default=3)
    ap.add_argument('--event-overlay-render-mode', choices=['bgr', 'mono_sparse'], default='bgr')
    ap.add_argument('--event-overlay-mono-color', choices=['white', 'green', 'cyan'], default='white')
    ap.add_argument('--event-overlay-mono-decay-us', type=int, default=0)
    ap.add_argument('--event-overlay-draw-debug-text', action='store_true', default=True)
    ap.add_argument('--disable-event-overlay-draw-debug-text', dest='event_overlay_draw_debug_text', action='store_false')
    ap.add_argument('--event-tsf1-pub-max-fps', type=float, default=170.0)
    ap.add_argument('--event-obs-submit-max-fps', type=float, default=170.0)
    ap.add_argument('--event-viz-after-human-mask-filter', dest='event_viz_after_human_mask_filter', action='store_true', default=True)
    ap.add_argument('--disable-event-viz-after-human-mask-filter', dest='event_viz_after_human_mask_filter', action='store_false')
    ap.add_argument('--event-viz-buffer-enable', dest='event_viz_buffer_enable', action='store_true', default=True)
    ap.add_argument('--disable-event-viz-buffer', dest='event_viz_buffer_enable', action='store_false')
    ap.add_argument('--event-viz-buffer-ms', type=float, default=50.0)
    ap.add_argument('--event-viz-buffer-slots', type=int, default=0)
    ap.add_argument('--event-viz-buffer-drop-policy', default='drop_oldest')
    ap.add_argument('--event-viz-max-output-age-ms', type=float, default=50.0)
    ap.add_argument('--event-viz-build-max-fps', type=float, default=0.0)
    ap.add_argument('--event-viz-skip-empty-fastpath', dest='event_viz_skip_empty_fastpath', action='store_true', default=False)
    ap.add_argument('--disable-event-viz-skip-empty-fastpath', dest='event_viz_skip_empty_fastpath', action='store_false')
    ap.add_argument('--event-viz-worker-mode', default='thread', choices=['thread', 'off'])
    ap.add_argument('--event-viz-no-mask-policy', default='empty', choices=['empty', 'filtered', 'raw_debug', 'skip'])
    ap.add_argument('--event-viz-idle-heartbeat-fps', type=float, default=20.0)
    ap.add_argument(
        '--event-track-input-policy',
        default='post_mask',
        choices=['post_mask', 'raw', 'raw_on_mask_unavailable', 'raw_when_filtered_empty', 'raw_on_mask_unavailable_or_filtered_empty'],
        help='Forward to event workers: STC/spatter/line_filter input source relative to human-mask filtering.',
    )
    ap.add_argument('--event-obs-sidecar-enable', dest='event_obs_sidecar_enable', action='store_true', default=True)
    ap.add_argument('--disable-event-obs-sidecar', dest='event_obs_sidecar_enable', action='store_false')
    ap.add_argument('--event-overlay-shm-schema-version', type=int, default=2)
    ap.add_argument('--event-worker-status-enable', action='store_true', default=True)
    ap.add_argument('--disable-event-worker-status', dest='event_worker_status_enable', action='store_false')
    ap.add_argument('--event-worker-status-interval-ms', type=float, default=1000.0)
    ap.add_argument('--event-hot-pixel-filter-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-hot-pixel-filter', dest='event_hot_pixel_filter_enable', action='store_false')
    ap.add_argument('--event-hot-pixel-filter-mode', default='tracking_only')
    ap.add_argument('--event-hot-pixel-radius-px', type=int, default=0)
    ap.add_argument('--event-hot-pixel-static-map', default='')
    ap.add_argument('--event-hot-pixel-learn-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-hot-pixel-learn', dest='event_hot_pixel_learn_enable', action='store_false')
    ap.add_argument('--event-hot-pixel-learn-min-fraction', type=float, default=0.25)
    ap.add_argument('--event-hot-pixel-learn-min-count', type=int, default=64)
    ap.add_argument('--event-hot-pixel-learn-confirm-slices', type=int, default=5)
    ap.add_argument('--event-hot-pixel-learn-max-pixels', type=int, default=16)
    ap.add_argument('--event-cluster-sanitize-enable', action='store_true', default=True)
    ap.add_argument('--disable-event-cluster-sanitize', dest='event_cluster_sanitize_enable', action='store_false')
    ap.add_argument('--event-line-shadow-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-line-shadow', dest='event_line_shadow_enable', action='store_false')
    ap.add_argument('--event-line-shadow-cameras', default='')
    ap.add_argument('--event-line-shadow-min-step-px', type=float, default=-1.0)
    ap.add_argument('--event-line-shadow-min-total-disp-px', type=float, default=-1.0)
    ap.add_argument('--event-line-shadow-bullet-min-output-streak', type=int, default=-1)
    ap.add_argument('--event-line-shadow-untracked-threshold', type=int, default=-1)
    ap.add_argument('--event-line-shadow-line-min-valid-step-count', type=int, default=-1)
    ap.add_argument('--active-led-marker-enable', action='store_true', default=False,
                    help='Run the default-off IR-ID Active LED decoder on one explicitly selected Event camera.')
    ap.add_argument('--disable-active-led-marker', dest='active_led_marker_enable', action='store_false')
    ap.add_argument('--active-led-marker-cameras', default='',
                    help='Exactly one Event camera alias allowed for the minimum IR-ID field run, e.g. G.')
    ap.add_argument('--active-led-marker-roi', default='',
                    help='Required marker event ROI x,y,w,h for the selected IR-ID camera.')
    ap.add_argument('--active-led-marker-run-id', default='',
                    help='Required immutable run identifier written into every IR-ID observation.')
    ap.add_argument('--active-led-marker-queue-size', type=int, default=512)
    ap.add_argument('--active-led-marker-search-margin-px', type=int, default=12)
    ap.add_argument('--active-led-marker-stale-ms', type=float, default=1000.0)
    ap.add_argument('--active-led-marker-alpha-ms', type=float, default=20.0)
    ap.add_argument('--active-led-marker-bit-count', type=int, default=8)
    ap.add_argument('--active-led-marker-checksum', action='store_true', default=False)
    ap.add_argument('--active-led-marker-tolerance', type=float, default=0.35)
    ap.add_argument('--active-led-marker-bin-us', type=int, default=1000)
    ap.add_argument('--active-led-marker-min-events-per-bin', type=int, default=8)
    ap.add_argument('--active-led-marker-min-segment-events', type=int, default=20)
    ap.add_argument('--active-led-marker-merge-gap-us', type=int, default=2500)
    ap.add_argument('--active-led-marker-polarity', choices=['any', 'positive', 'negative', 'pos', 'neg'], default='any')
    ap.add_argument('--event-sequential-start-ready-gate-enable', action='store_true', default=False, help='Start EVENT workers one by one and wait for each ready flag/status before the next worker.')
    ap.add_argument('--disable-event-sequential-start-ready-gate', dest='event_sequential_start_ready_gate_enable', action='store_false')
    ap.add_argument('--event-sequential-start-timeout-sec', type=float, default=8.0)
    ap.add_argument('--event-sequential-start-poll-sec', type=float, default=0.2)
    ap.add_argument('--event-sequential-start-settle-sec', type=float, default=0.0)
    ap.add_argument('--event-sequential-start-continue-on-failure', action='store_true', default=True)
    ap.add_argument('--event-sequential-start-stop-on-failure', dest='event_sequential_start_continue_on_failure', action='store_false')
    ap.add_argument('--event-worker-human-mask-filter-enable', dest='event_worker_human_mask_filter_enable', action='store_true', default=True)
    ap.add_argument('--disable-event-worker-human-mask-filter', dest='event_worker_human_mask_filter_enable', action='store_false')
    ap.add_argument('--event-mask-evidence-enable', action='store_true', default=False, help='Run independent human-mask evidence ticker outside Event raw loop.')
    ap.add_argument('--disable-event-mask-evidence', dest='event_mask_evidence_enable', action='store_false')
    ap.add_argument('--event-mask-evidence-period-ms', type=float, default=16.0)
    ap.add_argument('--event-mask-evidence-stale-ms', type=float, default=120.0)
    ap.add_argument('--event-mask-evidence-tail-bytes', type=int, default=4000000)
    ap.add_argument('--event-mask-evidence-cpus', default='', help='Optional CPU list for independent mask evidence ticker.')
    ap.add_argument('--event-mask-evidence-control-json', default='sync_ipc/event_mask_control.json')
    ap.add_argument('--event-mask-evidence-status-json', default='sync_ipc/event_mask_evidence_status.json')
    ap.add_argument('--event-mask-evidence-json-template', default='sync_ipc/event_mask_evidence_{camera}.json')
    ap.add_argument('--event-mask-raster-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-mask-raster', dest='event_mask_raster_enable', action='store_false')
    ap.add_argument('--event-mask-raster-template', default='/dev/shm/event_human_mask_raster_{camera}.mmap')
    ap.add_argument('--event-mask-raster-width', type=int, default=1280)
    ap.add_argument('--event-mask-raster-height', type=int, default=720)
    ap.add_argument('--event-mask-raster-dilate-px', type=int, default=8)
    ap.add_argument('--event-mask-raster-erode-px', type=int, default=0)
    ap.add_argument('--event-mask-raster-min-conf', type=float, default=0.2)
    ap.add_argument('--event-loop-profiler-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-loop-profiler', dest='event_loop_profiler_enable', action='store_false')
    ap.add_argument('--event-loop-profiler-interval-ms', type=float, default=1000.0)
    ap.add_argument('--event-loop-profiler-jsonl-template', default='sync_ipc/event_loop_profile_{camera}.jsonl')
    ap.add_argument('--event-native-hal-broker-enable', action='store_true', default=False, help='Use one C++ HAL FIFO broker for all event cameras.')
    ap.add_argument('--disable-event-native-hal-broker', dest='event_native_hal_broker_enable', action='store_false')
    ap.add_argument('--event-source-mode', choices=['live_hal', 'file_replay'], default='live_hal',
                    help='Event source selected before startup. live_hal keeps the production HAL broker; file_replay feeds the same EVB FIFO from recorded RAW files.')
    ap.add_argument('--dataset-dir', default='',
                    help='V25 dataset folder selected before startup. Used to resolve Event/MVS replay sources and write source_mode_effective evidence.')
    ap.add_argument('--event-file-fifo-broker-bin', default='')
    ap.add_argument('--event-file-fifo-broker-raw-files', default='',
                    help='Comma-separated replay RAW mapping, e.g. A=/data/A.raw,B=/data/B.raw. Required when --event-source-mode=file_replay.')
    ap.add_argument('--event-file-fifo-broker-replay-timing', choices=['realtime', 'fast'], default='realtime')
    ap.add_argument('--event-file-fifo-broker-allow-missing-ext-trigger', action='store_true', default=False,
                    help='Debug only: allow replay RAW without External Trigger events. Default is fail-fast.')
    ap.add_argument('--event-file-fifo-broker-loop', action='store_true', default=False,
                    help='Loop Event RAW replay until launcher shutdown. Intended for full replay smoke/regression.')
    ap.add_argument('--event-file-fifo-broker-no-loop', dest='event_file_fifo_broker_loop', action='store_false')
    ap.add_argument('--event-native-hal-broker-bin', default='')
    ap.add_argument('--event-native-hal-broker-auto-build', action='store_true', default=True)
    ap.add_argument('--disable-event-native-hal-broker-auto-build', dest='event_native_hal_broker_auto_build', action='store_false')
    ap.add_argument('--event-native-hal-broker-build-dir', default='native_event_bridge/build')
    ap.add_argument('--event-native-hal-broker-out-dir', default='sync_ipc/native_hal_fifos')
    ap.add_argument('--event-native-hal-broker-fifo-open-mode', choices=['rdwr', 'blocking', 'nonblock'], default='rdwr')
    ap.add_argument('--event-native-hal-broker-duration-sec', type=int, default=0, help='0 means run until launcher stops it.')
    ap.add_argument('--event-native-hal-broker-slice-us', type=int, default=4000)
    ap.add_argument('--event-native-hal-broker-wait-mode', choices=['poll', 'blocking'], default='poll')
    ap.add_argument('--event-native-hal-broker-poll-sleep-us', type=int, default=100)
    ap.add_argument('--event-native-hal-broker-report-ms', type=int, default=1000)
    ap.add_argument('--event-native-hal-broker-open-retries', type=int, default=1)
    ap.add_argument('--event-native-hal-broker-open-retry-delay-ms', type=int, default=0)
    ap.add_argument('--event-native-hal-broker-stats-jsonl', default='sync_ipc/event_native_hal_broker_stats.jsonl')
    ap.add_argument('--event-native-hal-broker-stdout-log', default='sync_ipc/launcher_logs/event_native_hal_broker.out')
    ap.add_argument('--event-native-hal-broker-stderr-log', default='sync_ipc/launcher_logs/event_native_hal_broker.err')
    ap.add_argument('--event-native-hal-broker-startup-timeout-sec', type=float, default=30.0)
    ap.add_argument('--event-native-hal-broker-cpus', default='')
    ap.add_argument('--event-visual-mask-overlay-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-visual-mask-overlay', dest='event_visual_mask_overlay_enable', action='store_false')
    ap.add_argument('--event-visual-mask-raster-template', default='/dev/shm/event_human_mask_raster_{camera}.mmap')
    ap.add_argument('--event-visual-mask-overlay-name-template', default='event_overlay_masked_latest_{camera}')
    ap.add_argument('--event-visual-mask-overlay-sidecar-template', default='sync_ipc/event_overlay_masked_{camera}_latest.json')
    ap.add_argument('--event-visual-mask-overlay-history-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-visual-mask-overlay-history', dest='event_visual_mask_overlay_history_enable', action='store_false')
    ap.add_argument('--event-visual-mask-overlay-history-name-template', default='event_overlay_masked_hist_{camera}_{slot}')
    ap.add_argument('--event-visual-mask-overlay-history-sidecar-template', default='sync_ipc/event_overlay_masked_{camera}_history.json')
    ap.add_argument('--event-visual-mask-overlay-history-slots', type=int, default=64)
    ap.add_argument('--event-visual-mask-overlay-sparse-enable', action='store_true', default=False)
    ap.add_argument('--disable-event-visual-mask-overlay-sparse', dest='event_visual_mask_overlay_sparse_enable', action='store_false')
    ap.add_argument('--event-visual-mask-overlay-sparse-name-template', default='event_overlay_sparse_hist_{camera}_{slot}')
    ap.add_argument('--event-visual-mask-overlay-sparse-sidecar-template', default='sync_ipc/event_overlay_sparse_{camera}_history.json')
    ap.add_argument('--event-visual-mask-overlay-sparse-slots', type=int, default=64)
    ap.add_argument('--event-visual-mask-overlay-sparse-max-points', type=int, default=20000)
    ap.add_argument('--event-sync-map-ring-records', type=int, default=512)
    ap.add_argument('--event-sync-map-ring-write-interval-ms', type=float, default=100.0)
    ap.add_argument('--event-visual-mask-no-mask-policy', choices=['empty', 'hold_last_good', 'raw_debug', 'raw'], default='hold_last_good')
    ap.add_argument('--event-visual-mask-slice-us', type=int, default=33333)
    ap.add_argument('--event-visual-mask-accumulation-us', type=int, default=33333)
    ap.add_argument('--event-visual-mask-publish-fps', type=int, default=40)
    ap.add_argument('--event-visual-mask-stale-ms', type=int, default=120)
    ap.add_argument('--mvs-source-mode', choices=['live_camera', 'file_replay'], default='live_camera',
                    help='MVS visible-light source selected before startup. live_camera starts hardware workers; file_replay publishes recorded MVS frames to the same shm contracts.')
    ap.add_argument('--mvs-file-frame-broker-dataset-dir', default='',
                    help='Dataset folder containing recordings/<session>/mvs/camera*/video/index files. Required when --mvs-source-mode=file_replay.')
    ap.add_argument('--mvs-file-frame-broker-replay-timing', choices=['realtime', 'fast'], default='realtime')
    ap.add_argument('--mvs-file-frame-broker-max-frames', type=int, default=0)
    ap.add_argument('--mvs-file-frame-broker-loop', action='store_true', default=False)
    ap.add_argument('--mvs-file-frame-broker-no-loop', dest='mvs_file_frame_broker_loop', action='store_false')
    ap.add_argument('--mvs-file-frame-broker-report-ms', type=int, default=1000)
    ap.add_argument('--mvs-file-frame-broker-cpus', default='', help='Optional CPU set for the MVS file replay broker.')
    ap.add_argument('--mvs-aggregator-delay', type=float, default=0.8, help='Delay after starting coordinator before starting MVS/YOLO aggregator.')
    ap.add_argument('--mvs-worker-cpus', default='', help='Optional comma-separated CPU cores for workers, e.g. 3,4,5,6,7,8,9,10.')
    ap.add_argument('--mvs-coordinator-cpu', default='', help='Optional CPU core for trigger coordinator, e.g. 2.')
    ap.add_argument('--mvs-yolo-shard-workers-enable', action='store_true', help='Run multiple YOLO/MVS_AGG shard workers instead of one monolithic YOLO process.')
    ap.add_argument('--mvs-yolo-shards', default='', help='YOLO shard camera groups, e.g. A,C:B,D:E,G:F,H. Empty uses A,C:B,D:E,G:F,H for matching active cams.')
    ap.add_argument('--yolo-worker-cpus', default='', help='Optional comma-separated CPU sets for YOLO shard workers, e.g. 20-21,22-23,24-25,26-27.')
    ap.add_argument('--yolo-worker-omp-num-threads', type=int, default=1, help='OMP/MKL/OpenBLAS/NumExpr threads for each YOLO shard worker')
    ap.add_argument('--yolo-async-jsonl-writer-enable', action='store_true', default=True, help='Enable background human_result JSONL writers in YOLO workers (default: enabled).')
    ap.add_argument('--no-yolo-async-jsonl-writer', dest='yolo_async_jsonl_writer_enable', action='store_false', help='Disable background human_result JSONL writers in YOLO workers.')
    ap.add_argument('--yolo-async-jsonl-queue', type=int, default=512)
    ap.add_argument('--yolo-async-jsonl-queue-size', dest='yolo_async_jsonl_queue', type=int, help='Alias for --yolo-async-jsonl-queue')
    ap.add_argument('--yolo-async-jsonl-flush-ms', type=float, default=50.0)
    ap.add_argument('--yolo-postprocess-workers', type=int, default=2, help='Parallel per-image postprocess workers per YOLO shard. 0 keeps config value.')
    ap.add_argument('--yolo-external-postprocess-enable', action='store_true', help='Move YOLO stable-id/ReID/JSONL postprocess into one child process per YOLO shard.')
    ap.add_argument('--yolo-external-postprocess-workers', type=int, default=1, help='Requested external postprocess processes per shard; stateful path currently uses 1.')
    ap.add_argument('--yolo-external-postprocess-cpus', default='', help='Comma-separated CPU sets for external postprocess per YOLO shard, e.g. 10-11,12-13,14-15,16-17.')
    ap.add_argument('--yolo-external-postprocess-queue-size', type=int, default=4, help='Queued YOLO batches per shard before dropping oldest external postprocess work.')
    ap.add_argument('--yolo-external-postprocess-start-method', choices=['fork', 'spawn', 'forkserver'], default='', help='Multiprocessing start method for external YOLO postprocess. Linux default is fork.')
    ap.add_argument('--yolo-drop-stale-input-ms', type=float, default=0.0, help='Skip YOLO input frames older than this wall age. 0 disables.')
    ap.add_argument('--yolo-drop-stale-result-ms', type=float, default=0.0, help='Mark stale human results over this capture-to-result age. 0 disables.')
    ap.add_argument('--yolo-worker-status-interval-ms', type=float, default=500.0)
    ap.add_argument('--global-yolo-enable', action='store_true', help='Launch one global YOLO inference server that reads all split-MVS shm frames.')
    ap.add_argument('--global-yolo-shadow-mode', dest='global_yolo_shadow_mode', action='store_true', help='Run Global YOLO beside legacy shards and write results under --global-yolo-shadow-output-dir.')
    ap.add_argument('--global-yolo-no-shadow-mode', '--global-yolo-primary-mode', dest='global_yolo_shadow_mode', action='store_false', help='Run Global YOLO as the primary YOLO chain and skip legacy shard workers.')
    ap.add_argument('--global-yolo-cpus', default='', help='Optional CPU set for the global YOLO process, e.g. 20-27.')
    ap.add_argument('--global-yolo-primary-workers', type=int, default=1, help='Number of Global YOLO primary worker processes. 1 keeps the legacy single-process Global YOLO path.')
    ap.add_argument('--global-yolo-primary-groups', default='', help='Camera groups for multi-process Global YOLO, e.g. A,C,E,G:B,D,F,H. Empty uses --global-yolo-microbatch-groups.')
    ap.add_argument('--global-yolo-primary-cpus', default='', help='Comma-separated CPU sets for Global YOLO primary workers, e.g. 20-23,24-27.')
    ap.add_argument('--global-yolo-merge-enable', action='store_true', default=True, help='Merge multi-worker Global YOLO latest/status into the canonical global_yolo files.')
    ap.add_argument('--global-yolo-no-merge', dest='global_yolo_merge_enable', action='store_false')
    ap.add_argument('--global-yolo-merge-cpus', default='', help='Optional CPU set for the Global YOLO merger process.')
    ap.add_argument('--global-yolo-merge-poll-ms', type=float, default=20.0)
    ap.add_argument('--global-yolo-merge-worker-stale-ms', type=float, default=500.0)
    ap.add_argument('--global-yolo-model', default='', help='Override YOLO model path for Global YOLO. Empty uses MVS config model.model.')
    ap.add_argument('--global-yolo-engine', default='', help='Optional TensorRT/engine path for Global YOLO. When set it takes priority over --global-yolo-model.')
    ap.add_argument('--global-yolo-device', default='0')
    ap.add_argument('--global-yolo-imgsz', type=int, default=768)
    ap.add_argument('--global-yolo-conf', type=float, default=0.18)
    ap.add_argument('--global-yolo-iou', type=float, default=0.78)
    ap.add_argument('--global-yolo-max-det', type=int, default=20)
    ap.add_argument('--global-yolo-half', action='store_true', default=True)
    ap.add_argument('--global-yolo-no-half', dest='global_yolo_half', action='store_false')
    ap.add_argument('--global-yolo-retina-masks', action='store_true', default=False)
    ap.add_argument('--global-yolo-target-fps', type=float, default=80.0)
    ap.add_argument('--global-yolo-microbatch-groups', default='A,C,E,G:B,D,F,H', help='Global YOLO microbatch groups, e.g. A,C,E,G:B,D,F,H or A,B,C,D,E,F,G,H.')
    ap.add_argument('--global-yolo-batch-collect-wait-ms', type=float, default=0.0, help='After the first fresh frame in a Global YOLO group, wait briefly for missing cameras before inference.')
    ap.add_argument('--global-yolo-primary-batch-collect-wait-ms', default='', help='Comma-separated collect wait ms per Global YOLO primary worker, e.g. 2.0,4.0. Empty uses --global-yolo-batch-collect-wait-ms for every worker.')
    ap.add_argument('--global-yolo-batch-collect-retry-sleep-ms', type=float, default=0.5, help='Sleep interval while collecting missing Global YOLO frames.')
    ap.add_argument('--global-yolo-postprocess-backend', choices=['gpu_fast', 'torch_fast', 'cpu_legacy'], default='gpu_fast')
    ap.add_argument('--global-yolo-footpoint-mode', choices=['center', 'bottom', 'top'], default='center')
    ap.add_argument('--global-yolo-anchor-mode', choices=['bbox_center', 'bbox_bottom', 'mask_robust_center', 'mask_eroded_centroid', 'auto_topdown_center'], default='auto_topdown_center')
    ap.add_argument('--global-yolo-anchor-height-correction-enable', action='store_true', default=True)
    ap.add_argument('--global-yolo-no-anchor-height-correction-enable', dest='global_yolo_anchor_height_correction_enable', action='store_false')
    ap.add_argument('--global-yolo-anchor-height-correction-k', type=float, default=0.08)
    ap.add_argument('--global-yolo-anchor-nadir-pixel-template', default='calib_true/nadir_camera{camera}.json')
    ap.add_argument('--global-yolo-include-mask-polygons', action='store_true', default=False)
    ap.add_argument('--global-yolo-mask-polygon-max-points', type=int, default=96)
    ap.add_argument('--global-yolo-write-legacy-jsonl', action='store_true', default=True)
    ap.add_argument('--global-yolo-no-write-legacy-jsonl', dest='global_yolo_write_legacy_jsonl', action='store_false')
    ap.add_argument('--global-yolo-shadow-output-dir', default='sync_ipc/global_yolo_shadow')
    ap.add_argument('--global-yolo-latest-json', default='sync_ipc/global_yolo_latest.json')
    ap.add_argument('--global-yolo-status-json', default='sync_ipc/global_yolo_status.json')
    ap.add_argument('--global-yolo-perf-csv', default='sync_ipc/global_yolo_infer_perf.csv')
    ap.add_argument('--global-yolo-exposure-audit-csv', default='sync_ipc/global_yolo_exposure_audit.csv')
    ap.add_argument('--global-yolo-latest-interval-ms', type=float, default=100.0)
    ap.add_argument('--global-yolo-status-interval-ms', type=float, default=1000.0)
    ap.add_argument('--global-yolo-perf-flush-interval-ms', type=float, default=1000.0)
    ap.add_argument('--global-yolo-async-record-postprocess-enable', action='store_true', default=False, help='V24-Z: run Global YOLO polygon/anchor/record publication on a background thread.')
    ap.add_argument('--global-yolo-no-async-record-postprocess', dest='global_yolo_async_record_postprocess_enable', action='store_false')
    ap.add_argument('--global-yolo-async-record-postprocess-queue-size', type=int, default=2)
    ap.add_argument('--global-yolo-async-record-postprocess-policy', choices=['block'], default='block')
    ap.add_argument('--global-yolo-input-alignment-audit-enable', action='store_true', default=False, help='V24-Z3a: audit visible-frame sync/exposure alignment without changing Global YOLO batching.')
    ap.add_argument('--global-yolo-no-input-alignment-audit', dest='global_yolo_input_alignment_audit_enable', action='store_false')
    ap.add_argument('--global-yolo-input-alignment-audit-max-wait-ms', type=float, default=8.0)
    ap.add_argument('--global-yolo-input-alignment-audit-max-spread-ms', type=float, default=6.0)
    ap.add_argument('--global-yolo-input-alignment-audit-history', type=int, default=512)
    ap.add_argument('--global-yolo-input-buffer-enable', action='store_true', default=False, help='V24-Z4: align Global YOLO input via per-camera ring buffers by sync_index and exposure_mid.')
    ap.add_argument('--global-yolo-no-input-buffer', dest='global_yolo_input_buffer_enable', action='store_false')
    ap.add_argument('--global-yolo-input-buffer-size', type=int, default=12)
    ap.add_argument('--global-yolo-input-buffer-max-wait-ms', type=float, default=30.0)
    ap.add_argument('--global-yolo-input-buffer-max-spread-ms', type=float, default=6.0)
    ap.add_argument('--global-yolo-input-buffer-exposure-end-barrier-enable', action='store_true', default=True)
    ap.add_argument('--global-yolo-no-input-buffer-exposure-end-barrier', dest='global_yolo_input_buffer_exposure_end_barrier_enable', action='store_false')
    ap.add_argument('--clean-shm', action='store_true', help='Remove stale /dev/shm frames before start')
    ap.add_argument('--clean-jsonl', action='store_true', help='Remove old human/hit debug jsonl files before start')
    ap.add_argument('--ready-timeout', type=float, default=30.0)
    ap.add_argument('--event-ready-continue-on-timeout', action='store_true', help='Debug only: continue launching even if not all fresh event_ready flags appear before --ready-timeout.')
    ap.add_argument('--event-ready-usb-enable', action='store_true', default=False, help='Use event camera USB throughput as an additional startup ready signal.')
    ap.add_argument('--disable-event-ready-usb', dest='event_ready_usb_enable', action='store_false')
    ap.add_argument('--event-ready-usb-sudo-enable', action='store_true', default=False, help='Debug only: try sudo -n for USB throughput monitor. Default disabled to avoid password prompts in launch flow.')
    ap.add_argument('--event-ready-usb-threshold-mbps', type=float, default=0.3, help='Per-camera USB throughput threshold in MB/s for event startup ready.')
    ap.add_argument('--event-ready-usb-interval-sec', type=float, default=0.5, help='USB throughput sampling interval for event startup ready.')
    ap.add_argument('--event-ready-tolerate-missing-enable', action='store_true', default=True, help='Continue launching when only part of event cameras become ready.')
    ap.add_argument('--disable-event-ready-tolerate-missing', dest='event_ready_tolerate_missing_enable', action='store_false')
    ap.add_argument('--event-ready-min-ready-cams', type=int, default=1, help='Minimum event cameras required before degraded startup may continue. Set 0 to always continue after partial wait.')
    ap.add_argument('--event-ready-partial-wait-sec', type=float, default=8.0, help='Wait at least this long for event cameras before degraded startup may continue.')
    ap.add_argument('--event-startup-diagnostics-enable', action='store_true', default=True, help='Write EVENT startup diagnostics JSON while waiting for cameras.')
    ap.add_argument('--disable-event-startup-diagnostics', dest='event_startup_diagnostics_enable', action='store_false')
    ap.add_argument('--event-startup-diagnostics-window-enable', action='store_true', default=False, help='Show a temporary EVENT startup diagnostics window.')
    ap.add_argument('--disable-event-startup-diagnostics-window', dest='event_startup_diagnostics_window_enable', action='store_false')
    ap.add_argument('--event-startup-diagnostics-json', default='sync_ipc/startup_diagnostics.json')
    ap.add_argument('--event-startup-diagnostics-auto-exit-ok-ms', type=int, default=2500, help='Startup diagnostics window auto-closes this long after all cameras are ready.')
    ap.add_argument('--event-overlay-overview-process-enable', action='store_true', default=False, help='Run Event Overlay Overview in a standalone process instead of the main preview Qt loop.')
    ap.add_argument('--disable-event-overlay-overview-process', dest='event_overlay_overview_process_enable', action='store_false')
    ap.add_argument('--event-overlay-overview-cpus', default='', help='Optional taskset CPU list for standalone Event Overlay Overview.')
    ap.add_argument('--event-overlay-overview-status-json', default='sync_ipc/event_overlay_overview_status.json')
    ap.add_argument('--event-overlay-overview-control-json', default='sync_ipc/event_overlay_overview_control.json')
    ap.add_argument('--event-overlay-overview-status-interval-ms', type=float, default=1000.0)
    ap.add_argument('--event-overlay-window-label-fps', type=float, default=10.0)
    ap.add_argument('--mvs-delay', type=float, default=0.8)
    ap.add_argument('--hit-delay', type=float, default=0.8)
    ap.add_argument('--gl-delay', type=float, default=1.0)
    ap.add_argument(
        '--console-log-mode',
        choices=['all', 'startup', 'errors', 'quiet'],
        default='startup',
        help=(
            'Child process console output mode. '
            'all=print everything; startup=print first lines and important messages; '
            'errors=only important messages; quiet=no child output.'
        ),
    )
    ap.add_argument(
        '--startup-log-lines',
        type=int,
        default=60,
        help='How many startup lines to print per child process when --console-log-mode=startup.',
    )
    ap.add_argument(
        '--child-log-dir',
        default='sync_ipc/launcher_logs',
        help='Directory for full child process logs.',
    )
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--launch-profile', default='')
    pre.add_argument('--project-root', default='')
    pre_args, _ = pre.parse_known_args()
    applied_profile = _apply_launch_profile_defaults(ap, pre_args.launch_profile, pre_args.project_root) if pre_args.launch_profile else Path()
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parent
    os.chdir(project_root)
    print(f"[launcher] project_root={project_root}", flush=True)
    if applied_profile:
        print(f"[launcher] launch_profile={applied_profile}", flush=True)

    if args.config:
        unified_path = Path(args.config).expanduser()
        if not unified_path.is_absolute():
            unified_path = project_root / unified_path
        unified_cfg = load_json(unified_path)
        if not _is_unified_fusion_config(unified_cfg):
            raise SystemExit(f"[launcher][ERROR] --config must contain event_config and mvs_config sections: {unified_path}")
        event_cfg_path, mvs_cfg_path, event_cfg = _write_runtime_split_configs(unified_path, unified_cfg, project_root)
        args.event_config = str(event_cfg_path)
        args.mvs_config = str(mvs_cfg_path)
    else:
        event_cfg_path = project_root / args.event_config
        mvs_cfg_path = project_root / args.mvs_config
        event_cfg = load_json(event_cfg_path)
        _inject_event_sync_start_options(event_cfg, project_root)
    v25_source_mode_report, v25_resolved_manifest = _apply_v25_dataset_source_mode(args, project_root)
    cams = parse_cams(args.cams, event_cfg)
    cam_arg = ','.join(cams)
    print(f"[launcher] cams={cam_arg}", flush=True)
    if bool(args.active_led_marker_enable):
        marker_cameras = [
            item.strip().upper()
            for item in str(args.active_led_marker_cameras or '').replace(';', ',').split(',')
            if item.strip()
        ]
        available_cameras = {str(cam).strip().upper() for cam in cams}
        if len(marker_cameras) != 1:
            raise SystemExit('[launcher][ERROR] --active-led-marker-enable requires exactly one --active-led-marker-cameras alias')
        if marker_cameras[0] not in available_cameras:
            raise SystemExit(
                '[launcher][ERROR] IR-ID marker camera must be included in --cams; '
                f'requested={marker_cameras[0]} available={sorted(available_cameras)}'
            )
        if not str(args.active_led_marker_roi or '').strip():
            raise SystemExit('[launcher][ERROR] --active-led-marker-enable requires --active-led-marker-roi x,y,w,h')
        if not str(args.active_led_marker_run_id or '').strip():
            raise SystemExit('[launcher][ERROR] --active-led-marker-enable requires --active-led-marker-run-id')

    try:
        _patch_event_runtime_cpu_options(Path(args.event_config), project_root, args, cams)
    except Exception as _exc:
        print(f"[launcher][WARN] event CPU option patch skipped: {_exc}", flush=True)

    (project_root / 'sync_ipc').mkdir(exist_ok=True)
    if args.clean_shm or args.clean_jsonl:
        cleanup_stale(project_root, cams, clean_logs=args.clean_jsonl)
    initialize_event_mask_control_defaults(project_root, args)

    py = sys.executable or 'python3'

    child_log_dir = Path(args.child_log_dir).expanduser()
    if not child_log_dir.is_absolute():
        child_log_dir = project_root / child_log_dir
    child_log_dir.mkdir(parents=True, exist_ok=True)
    launcher_main_log = child_log_dir / 'launcher_main.log'

    def log_launcher(message: str, level: str = "INFO") -> None:
        line = f"[launcher][{level}] {message}"
        print(line, flush=True)
        try:
            with open(launcher_main_log, "a", encoding="utf-8", buffering=1) as fp:
                fp.write(line + "\n")
        except Exception:
            pass

    try:
        with open(launcher_main_log, "w", encoding="utf-8", buffering=1) as fp:
            fp.write(f"[launcher][INFO] project_root={project_root}\n")
            if applied_profile:
                fp.write(f"[launcher][INFO] launch_profile={applied_profile}\n")
            fp.write(f"[launcher][INFO] cams={cam_arg}\n")
    except Exception:
        pass
    source_mode_effective = _write_source_mode_effective(
        args,
        project_root,
        applied_profile,
        cams,
        v25_source_mode_report,
        v25_resolved_manifest,
    )
    log_launcher(
        "source_mode_effective "
        f"event={source_mode_effective.get('event_source_mode_effective')} "
        f"mvs={source_mode_effective.get('mvs_source_mode_effective')} "
        f"dataset={source_mode_effective.get('dataset_dir') or '-'}"
    )
    print(f"[launcher] child logs dir={child_log_dir}", flush=True)
    _ensure_default_mvs_env(log_launcher)
    print(
        f"[launcher] console_log_mode={args.console_log_mode} startup_log_lines={args.startup_log_lines}",
        flush=True,
    )
    event_serials = _event_serial_map(event_cfg, cams)
    startup_diag_path = Path(str(getattr(args, 'event_startup_diagnostics_json', 'sync_ipc/startup_diagnostics.json') or 'sync_ipc/startup_diagnostics.json')).expanduser()
    if not startup_diag_path.is_absolute():
        startup_diag_path = project_root / startup_diag_path
    _write_startup_diag(
        startup_diag_path if bool(getattr(args, 'event_startup_diagnostics_enable', True)) else None,
        state="initializing",
        title="Preparing EVENT startup",
        cams=cams,
        statuses={c: {"ready": False, "serial": event_serials.get(c, ""), "reason": "not started"} for c in cams},
        threshold_mbps=float(getattr(args, 'event_ready_usb_threshold_mbps', 0.3) or 0.3),
        message="launcher is preparing event camera process",
    )
    _startup_diag_proc = _start_startup_diag_window(project_root, py, startup_diag_path, args, log_launcher)

    if args.preview_backend == 'pyopengl-cuda':
        os.environ['PREVIEW_TRAJECTORY_SHM_ENABLE'] = '1'
        os.environ['PREVIEW_TRAJECTORY_SHM'] = str(args.preview_trajectory_shm)
        os.environ['PREVIEW_TRAJECTORY_LIFE_MS'] = str(args.preview_trajectory_life_ms)
        os.environ['PREVIEW_TRAJECTORY_MAX_SEGMENTS'] = str(max(1024, int(args.preview_trajectory_max_segments or 8192)))
        print(
            f"[launcher] preview_backend=pyopengl-cuda video_fps={args.preview_video_fps} render_fps={args.preview_render_fps} "
            f"trajectory_fps={args.preview_trajectory_fps} trajectory_life_ms={args.preview_trajectory_life_ms} "
            f"trajectory_max_segments={args.preview_trajectory_max_segments} "
            f"raw_template={args.preview_raw_template} trajectory_shm={args.preview_trajectory_shm}",
            flush=True,
        )

    event_script = resolve_script(project_root, ['multi_camera_launcher.py'])
    mvs_script = resolve_script(
        project_root,
        [
            'hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py',
            'cpp_bullet_core/rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py',
            'rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py',
        ],
    )
    hit_script = resolve_script(project_root, ['hit_judge_server.py'])
    gl_script = resolve_script(
        project_root,
        [
            'rentijiance/fusion_renderer_gl.py',
            'fusion_renderer_gl.py',
            'cpp_bullet_core/rentijiance/fusion_renderer_gl.py',
        ],
    )
    cuda_gl_script = resolve_script(project_root, ['pyopengl_cuda_preview_renderer.py'])
    event_overlay_overview_script = resolve_script(project_root, ['event_overlay_overview_renderer.py'])
    global_yolo_script = resolve_script(project_root, ['global_yolo_infer_server.py'])
    global_yolo_merge_script = resolve_script(project_root, ['global_yolo_merge_publisher.py'])
    global_bev_script = resolve_script(project_root, ['global_bev_fusion_renderer.py'])
    global_person_id_script = resolve_script(project_root, ['global_person_id_server.py'])
    person_id_dataset_recorder_script = resolve_script(project_root, ['person_id_dataset_recorder.py'])
    event_mask_evidence_script = resolve_script(project_root, ['event_mask_evidence_service.py'])
    global_bullet_reprocess_script = resolve_script(project_root, ['global_bullet_reprocess_service.py'])
    global_bullet_fusion_script = resolve_script(project_root, ['global_bullet_fusion_service.py'])
    damage_attribution_script = resolve_script(project_root, ['damage_attribution_service.py'])
    foam_dart_speed_script = resolve_script(project_root, ['foam_dart_speed_monitor_service.py'])
    manual_hit_script = resolve_script(project_root, ['manual_hit_service.py'])
    game_event_bridge_script = resolve_script(project_root, ['game_event_bridge_service.py'])
    trigger_coord_script = resolve_script(project_root, ['mvs_trigger_coordinator.py'])
    mvs_worker_script = resolve_script(project_root, ['mvs_cam_worker_external_trigger.py'])
    mvs_file_frame_broker_script = resolve_script(project_root, ['remote_ops/mvs_file_frame_broker.py'])

    procs: List[ManagedProc] = []
    stopping = False

    def make_proc(name: str, cmd: List[str], critical: bool = True, startup_lines: Optional[int] = None) -> ManagedProc:
        return ManagedProc(
            name,
            cmd,
            project_root,
            critical=critical,
            log_dir=child_log_dir,
            console_mode=args.console_log_mode,
            startup_lines=args.startup_log_lines if startup_lines is None else int(startup_lines),
        )

    def project_abs_path(value: str) -> str:
        p = Path(str(value or '')).expanduser()
        if p.is_absolute():
            return str(p)
        return str((project_root / p).resolve())

    def sync_abs_path(value: str) -> str:
        p = Path(str(value or '')).expanduser()
        if p.is_absolute():
            return str(p)
        normalized = str(value or '').replace('\\', '/')
        if normalized.startswith('sync_ipc/'):
            return str((project_root / p).resolve())
        return str((project_root / 'sync_ipc' / p).resolve())

    def start_mvs_file_frame_broker() -> None:
        dataset_dir = str(getattr(args, 'mvs_file_frame_broker_dataset_dir', '') or '').strip()
        if not dataset_dir:
            raise RuntimeError("mvs_source_mode=file_replay requires --mvs-file-frame-broker-dataset-dir")
        raw_preview_needed = bool(args.preview_backend == 'pyopengl-cuda' or args.global_bev_enable)
        slot_count = max(3, int(args.global_bev_ring_slots if args.global_bev_enable else 3))
        broker_cmd = [
            py, mvs_file_frame_broker_script,
            '--dataset-dir', project_abs_path(dataset_dir),
            '--sync-root', str((project_root / 'sync_ipc').resolve()),
            '--cams', cam_arg,
            '--replay-timing', str(getattr(args, 'mvs_file_frame_broker_replay_timing', 'realtime') or 'realtime'),
            '--raw-preview-template', str(args.preview_raw_template),
            '--raw-preview-slot-count', str(slot_count),
            '--status-json', str((project_root / 'sync_ipc' / 'mvs_file_frame_broker_status.json').resolve()),
            '--stats-jsonl', str((project_root / 'sync_ipc' / 'mvs_file_frame_broker_stats.jsonl').resolve()),
            '--report-ms', str(getattr(args, 'mvs_file_frame_broker_report_ms', 1000) or 1000),
            '--trigger-fps', str(float(args.mvs_trigger_fps or 40.0)),
        ]
        if not raw_preview_needed:
            broker_cmd += ['--no-raw-preview']
        if int(getattr(args, 'mvs_file_frame_broker_max_frames', 0) or 0) > 0:
            broker_cmd += ['--max-frames', str(int(args.mvs_file_frame_broker_max_frames))]
        if bool(getattr(args, 'mvs_file_frame_broker_loop', False)):
            broker_cmd += ['--loop']
        broker_cmd = _taskset_cmd(str(getattr(args, 'mvs_file_frame_broker_cpus', '') or ''), broker_cmd)
        log_launcher(
            f"MVS source=file_replay dataset={dataset_dir} timing={getattr(args, 'mvs_file_frame_broker_replay_timing', 'realtime')} "
            f"loop={bool(getattr(args, 'mvs_file_frame_broker_loop', False))} raw_preview={raw_preview_needed}"
        )
        p = make_proc('MVS_FILE_BROKER', broker_cmd, critical=True)
        procs.append(p); p.start()

    def start_global_yolo_if_enabled() -> None:
        if not bool(getattr(args, 'global_yolo_enable', False)):
            return

        def parse_worker_groups() -> List[List[str]]:
            worker_count = max(1, int(getattr(args, 'global_yolo_primary_workers', 1) or 1))
            if worker_count <= 1:
                return [list(cams)]
            raw = str(getattr(args, 'global_yolo_primary_groups', '') or '').strip()
            if not raw:
                raw = str(getattr(args, 'global_yolo_microbatch_groups', '') or '').strip()
            valid = set(cams)
            groups: List[List[str]] = []
            for part in raw.split(':'):
                group = [x.strip() for x in part.split(',') if x.strip()]
                group = [x for x in group if x in valid]
                if group:
                    groups.append(group)
            if not groups:
                groups = [list(cams[i::worker_count]) for i in range(worker_count)]
            used = {c for group in groups for c in group}
            missing = [c for c in cams if c not in used]
            if missing:
                if len(groups) < worker_count:
                    groups.append(missing)
                else:
                    groups[-1].extend(missing)
            groups = [g for g in groups if g]
            if len(groups) != worker_count:
                log_launcher(
                    f"global_yolo_primary_workers={worker_count} but parsed groups={len(groups)}; using parsed groups {groups}",
                    level="WARN",
                )
            return groups

        def parse_primary_collect_waits() -> List[float]:
            raw = str(getattr(args, 'global_yolo_primary_batch_collect_wait_ms', '') or '').strip()
            if not raw:
                return []
            waits: List[float] = []
            for part in raw.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    waits.append(max(0.0, float(part)))
                except Exception:
                    log_launcher(f"invalid global_yolo_primary_batch_collect_wait_ms item ignored: {part}", level="WARN")
            return waits

        primary_collect_waits = parse_primary_collect_waits()

        def build_worker_cmd(
            worker_index: int,
            worker_cams: List[str],
            latest_json: str,
            status_json: str,
            perf_csv: str,
            exposure_audit_csv: str,
            shadow_output_dir: str,
        ) -> List[str]:
            worker_cam_arg = ','.join(worker_cams)
            collect_wait_ms = float(args.global_yolo_batch_collect_wait_ms)
            if worker_index < len(primary_collect_waits):
                collect_wait_ms = primary_collect_waits[worker_index]
            gy_cmd = [
                py, global_yolo_script,
                '--project-root', str(project_root),
                '--config', str(args.mvs_config),
                '--cams', worker_cam_arg,
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--mvs-shm-template', 'mvs_latest_{camera}',
                '--mvs-meta-template', '{root}/mvs_latest_{camera}.json',
                '--calib-template', str(args.global_bev_calib_template),
                '--device', str(args.global_yolo_device),
                '--imgsz', str(args.global_yolo_imgsz),
                '--conf', str(args.global_yolo_conf),
                '--iou', str(args.global_yolo_iou),
                '--max-det', str(args.global_yolo_max_det),
                '--target-fps', str(args.global_yolo_target_fps),
                '--microbatch-groups', worker_cam_arg,
                '--batch-collect-wait-ms', str(collect_wait_ms),
                '--batch-collect-retry-sleep-ms', str(args.global_yolo_batch_collect_retry_sleep_ms),
                '--postprocess-backend', str(args.global_yolo_postprocess_backend),
                '--footpoint-mode', str(args.global_yolo_footpoint_mode),
                '--anchor-mode', str(args.global_yolo_anchor_mode),
                '--anchor-height-correction-k', str(args.global_yolo_anchor_height_correction_k),
                '--anchor-nadir-pixel-template', str(args.global_yolo_anchor_nadir_pixel_template),
                '--mask-polygon-max-points', str(args.global_yolo_mask_polygon_max_points),
                '--shadow-output-dir', shadow_output_dir,
                '--latest-json', latest_json,
                '--status-json', status_json,
                '--perf-csv', perf_csv,
                '--exposure-audit-csv', exposure_audit_csv,
                '--latest-interval-ms', str(args.global_yolo_latest_interval_ms),
                '--status-interval-ms', str(args.global_yolo_status_interval_ms),
                '--perf-flush-interval-ms', str(args.global_yolo_perf_flush_interval_ms),
                '--async-record-postprocess-queue-size', str(args.global_yolo_async_record_postprocess_queue_size),
                '--async-record-postprocess-policy', str(args.global_yolo_async_record_postprocess_policy),
                '--input-alignment-audit-max-wait-ms', str(args.global_yolo_input_alignment_audit_max_wait_ms),
                '--input-alignment-audit-max-spread-ms', str(args.global_yolo_input_alignment_audit_max_spread_ms),
                '--input-alignment-audit-history', str(args.global_yolo_input_alignment_audit_history),
                '--input-buffer-size', str(args.global_yolo_input_buffer_size),
                '--input-buffer-max-wait-ms', str(args.global_yolo_input_buffer_max_wait_ms),
                '--input-buffer-max-spread-ms', str(args.global_yolo_input_buffer_max_spread_ms),
            ]
            if str(args.global_yolo_model or '').strip():
                gy_cmd += ['--model', str(args.global_yolo_model)]
            if str(args.global_yolo_engine or '').strip():
                gy_cmd += ['--engine', str(args.global_yolo_engine)]
            if bool(args.global_yolo_half):
                gy_cmd += ['--half']
            else:
                gy_cmd += ['--no-half']
            if bool(args.global_yolo_retina_masks):
                gy_cmd += ['--retina-masks']
            if bool(args.global_yolo_anchor_height_correction_enable):
                gy_cmd += ['--anchor-height-correction-enable']
            else:
                gy_cmd += ['--no-anchor-height-correction-enable']
            if bool(args.global_yolo_include_mask_polygons):
                gy_cmd += ['--include-mask-polygons']
            if bool(getattr(args, 'global_yolo_async_record_postprocess_enable', False)):
                gy_cmd += ['--async-record-postprocess']
            if bool(getattr(args, 'global_yolo_input_alignment_audit_enable', False)):
                gy_cmd += ['--input-alignment-audit']
            if bool(getattr(args, 'global_yolo_input_buffer_enable', False)):
                gy_cmd += ['--input-buffer-enable']
            else:
                gy_cmd += ['--no-input-buffer']
            if bool(getattr(args, 'global_yolo_input_buffer_exposure_end_barrier_enable', True)):
                gy_cmd += ['--input-buffer-exposure-end-barrier']
            else:
                gy_cmd += ['--no-input-buffer-exposure-end-barrier']
            if bool(args.global_yolo_write_legacy_jsonl):
                gy_cmd += ['--write-legacy-jsonl']
            else:
                gy_cmd += ['--no-write-legacy-jsonl']
            if bool(args.global_yolo_shadow_mode):
                gy_cmd += ['--shadow-mode']
            cpus_raw = str(getattr(args, 'global_yolo_primary_cpus', '') or '').strip()
            if not cpus_raw:
                cpus_raw = str(getattr(args, 'global_yolo_cpus', '') or '').strip()
            return _taskset_cmd(_split_worker_cpu(cpus_raw, worker_index), gy_cmd)

        worker_groups = parse_worker_groups()
        use_multi_worker = len(worker_groups) > 1 and int(getattr(args, 'global_yolo_primary_workers', 1) or 1) > 1
        effective_collect_waits = [
            primary_collect_waits[i] if i < len(primary_collect_waits) else float(args.global_yolo_batch_collect_wait_ms)
            for i in range(len(worker_groups))
        ]
        primary_mode = not bool(args.global_yolo_shadow_mode)
        critical = bool(primary_mode)
        engine_enabled = bool(str(getattr(args, 'global_yolo_engine', '') or '').strip())
        mask_postprocess_enabled = bool(getattr(args, 'global_yolo_retina_masks', False) or getattr(args, 'global_yolo_include_mask_polygons', False))
        log_launcher(
            "Global YOLO mode={mode} workers={workers} groups={groups} collect_wait_ms={waits} merge={merge} "
            "legacy_yolo_skipped={legacy_skipped} postprocess_backend={backend} async_record_postprocess={async_record} "
            "input_alignment_audit={input_align_audit} input_buffer={input_buffer} postprocess_draw_enabled=False mask_postprocess_enabled={mask_post} "
            "tensorrt_enabled={trt}".format(
                mode="primary" if primary_mode else "shadow",
                workers=len(worker_groups),
                groups=worker_groups,
                waits=effective_collect_waits,
                merge=bool(getattr(args, 'global_yolo_merge_enable', True)) and use_multi_worker,
                legacy_skipped=primary_mode,
                backend=str(getattr(args, 'global_yolo_postprocess_backend', '')),
                async_record=bool(getattr(args, 'global_yolo_async_record_postprocess_enable', False)),
                input_align_audit=bool(getattr(args, 'global_yolo_input_alignment_audit_enable', False)),
                input_buffer=bool(getattr(args, 'global_yolo_input_buffer_enable', False)),
                mask_post=mask_postprocess_enabled,
                trt=engine_enabled,
            )
        )
        worker_latest_paths: List[str] = []
        worker_status_paths: List[str] = []

        if not use_multi_worker:
            gy_cmd = build_worker_cmd(
                0,
                list(cams),
                str(args.global_yolo_latest_json),
                str(args.global_yolo_status_json),
                str(args.global_yolo_perf_csv),
                str(args.global_yolo_exposure_audit_csv),
                str(args.global_yolo_shadow_output_dir),
            )
            p = make_proc('GLOBAL_YOLO', gy_cmd, critical=critical)
            procs.append(p); p.start()
            return

        log_launcher(f"starting {len(worker_groups)} Global YOLO primary workers: {worker_groups}")
        for idx, group in enumerate(worker_groups):
            latest_json = f"sync_ipc/global_yolo_worker_{idx}_latest.json"
            status_json = f"sync_ipc/global_yolo_worker_{idx}_status.json"
            perf_csv = f"sync_ipc/global_yolo_worker_{idx}_perf.csv"
            exposure_audit_csv = f"sync_ipc/global_yolo_worker_{idx}_exposure_audit.csv"
            shadow_dir = f"{str(args.global_yolo_shadow_output_dir).rstrip('/')}/worker_{idx}"
            worker_latest_paths.append(latest_json)
            worker_status_paths.append(status_json)
            gy_cmd = build_worker_cmd(idx, group, latest_json, status_json, perf_csv, exposure_audit_csv, shadow_dir)
            p = make_proc(f'GLOBAL_YOLO_{idx}', gy_cmd, critical=critical)
            procs.append(p); p.start()

        if bool(getattr(args, 'global_yolo_merge_enable', True)):
            merge_cmd = [
                py, global_yolo_merge_script,
                '--project-root', str(project_root),
                '--cams', cam_arg,
                '--worker-latest-jsons', ','.join(worker_latest_paths),
                '--worker-status-jsons', ','.join(worker_status_paths),
                '--latest-json', str(args.global_yolo_latest_json),
                '--status-json', str(args.global_yolo_status_json),
                '--poll-ms', str(args.global_yolo_merge_poll_ms),
                '--latest-interval-ms', str(args.global_yolo_latest_interval_ms),
                '--status-interval-ms', str(args.global_yolo_status_interval_ms),
                '--worker-stale-ms', str(args.global_yolo_merge_worker_stale_ms),
            ]
            merge_cmd = _taskset_cmd(str(getattr(args, 'global_yolo_merge_cpus', '') or ''), merge_cmd)
            p = make_proc('GLOBAL_YOLO_MERGE', merge_cmd, critical=critical)
            procs.append(p); p.start()
        else:
            log_launcher("multi-worker Global YOLO enabled without merger; canonical global_yolo_status/latest will not be updated", level="WARN")

    def stop_all(signum=None, frame=None):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        log_launcher("stopping all processes...")
        for mp in reversed(procs):
            mp.terminate()

    try:
        signal.signal(signal.SIGINT, stop_all)
        signal.signal(signal.SIGTERM, stop_all)
    except Exception:
        pass

    try:
        if not args.skip_event:
            cleanup_event_start_sync_files(project_root, cams)
            event_start_wall = time.time()
            p = make_proc('EVENT', [py, event_script, '--config', args.event_config], critical=True)
            procs.append(p); p.start()
            event_ready_ok = wait_ready(
                project_root / 'sync_ipc',
                cams,
                args.ready_timeout,
                not_before_wall=event_start_wall,
                project_root=project_root,
                serial_map=event_serials,
                usb_enable=bool(getattr(args, 'event_ready_usb_enable', False)),
                usb_threshold_mbps=float(getattr(args, 'event_ready_usb_threshold_mbps', 0.3) or 0.3),
                usb_interval_sec=float(getattr(args, 'event_ready_usb_interval_sec', 0.5) or 0.5),
                usb_sudo_enable=bool(getattr(args, 'event_ready_usb_sudo_enable', False)),
                allow_partial=bool(getattr(args, 'event_ready_tolerate_missing_enable', True)),
                min_ready_cams=int(getattr(args, 'event_ready_min_ready_cams', 1) or 0),
                partial_wait_sec=float(getattr(args, 'event_ready_partial_wait_sec', 8.0) or 0.0),
                diag_path=startup_diag_path if bool(getattr(args, 'event_startup_diagnostics_enable', True)) else None,
                log_cb=log_launcher,
            )
            if not event_ready_ok and not args.event_ready_continue_on_timeout:
                log_launcher(
                    "event ready timeout; aborting startup before MVS/YOLO/trigger. "
                    "Inspect sync_ipc/startup_diagnostics.json and sync_ipc/launcher_logs/event.log.",
                    "ERROR",
                )
                stop_all()
                return 2
            if args.event_ready_settle_delay > 0:
                time.sleep(args.event_ready_settle_delay)
        if args.mvs_delay > 0:
            time.sleep(args.mvs_delay)
        if not args.skip_mvs:
            mvs_source_mode = str(getattr(args, 'mvs_source_mode', 'live_camera') or 'live_camera').strip().lower()
            if mvs_source_mode == 'file_replay':
                start_mvs_file_frame_broker()
                if args.mvs_worker_ready_delay > 0:
                    time.sleep(args.mvs_worker_ready_delay)
                start_global_yolo_if_enabled()
                _write_yolo_chain_mode_status(
                    project_root, args, cams,
                    legacy_enabled=False,
                    reason="disabled_by_mvs_file_replay_source",
                )
                if not bool(getattr(args, 'global_yolo_enable', False)):
                    log_launcher(
                        "mvs_source_mode=file_replay is active but global_yolo_enable is false; "
                        "legacy monolithic MVS_AGG is intentionally not started to avoid opening live cameras.",
                        level="WARN",
                    )
                if args.mvs_aggregator_delay > 0:
                    time.sleep(args.mvs_aggregator_delay)
            elif args.mvs_split_workers:
                _enable_external_mvs_shm_mode(Path(args.mvs_config), project_root)
                worker_cpus = str(args.mvs_worker_cpus or '')
                raw_preview_needed = bool(args.preview_backend == 'pyopengl-cuda' or args.global_bev_enable)
                for idx, cid in enumerate(cams):
                    cpu = _split_worker_cpu(worker_cpus, idx)
                    worker_cmd = [py, mvs_worker_script, '--config', args.mvs_config, '--cam', str(cid)]
                    if raw_preview_needed:
                        worker_cmd += [
                            '--raw-preview-shm-enable',
                            '--raw-preview-shm-template', str(args.preview_raw_template),
                            '--raw-preview-slot-count', str(max(3, int(args.global_bev_ring_slots if args.global_bev_enable else 3))),
                        ]
                    worker_cmd = _taskset_cmd(cpu, worker_cmd)
                    p = make_proc(f'MVS_{cid}', worker_cmd, critical=True)
                    procs.append(p); p.start()
                if args.mvs_worker_ready_delay > 0:
                    time.sleep(args.mvs_worker_ready_delay)
                sync_start_file = str((project_root / 'sync_ipc' / 'sync_start.flag').resolve())
                coord_cmd = [
                    py, trigger_coord_script, '--config', args.mvs_config, '--cams', cam_arg,
                    '--lead-ms', str(args.mvs_trigger_lead_ms),
                    '--start-delay-ms', str(args.mvs_trigger_start_delay_ms),
                    '--sync-start-file', sync_start_file,
                    '--sync-start-settle-ms', str(args.mvs_trigger_sync_start_settle_ms),
                    '--status-file', str((project_root / 'sync_ipc' / 'trigger_status.json').resolve()),
                    '--timeline-file', str((project_root / 'sync_ipc' / 'global_trigger_timeline.jsonl').resolve()),
                    '--timeline-latest-file', str((project_root / 'sync_ipc' / 'global_trigger_timeline_latest.json').resolve()),
                ]
                if float(args.mvs_trigger_fps or 0.0) > 0.0:
                    coord_cmd += ['--fps', str(args.mvs_trigger_fps)]
                coord_cmd = _taskset_cmd(str(args.mvs_coordinator_cpu or ''), coord_cmd)
                p = make_proc('TRIGGER', coord_cmd, critical=True)
                procs.append(p); p.start()
                if args.mvs_aggregator_delay > 0:
                    time.sleep(args.mvs_aggregator_delay)
                start_global_yolo_if_enabled()
                run_legacy_yolo = not (bool(args.global_yolo_enable) and not bool(args.global_yolo_shadow_mode))
                legacy_reason = (
                    "disabled_by_global_yolo_primary"
                    if not run_legacy_yolo else
                    "enabled_because_global_yolo_disabled_or_shadow_mode"
                )
                _write_yolo_chain_mode_status(project_root, args, cams, run_legacy_yolo, legacy_reason)
                if not run_legacy_yolo:
                    log_launcher("legacy shard YOLO skipped because Global YOLO primary mode is active")
                if run_legacy_yolo and (args.mvs_yolo_shard_workers_enable or str(args.mvs_yolo_shards or '').strip()):
                    shards = _parse_yolo_shards(args.mvs_yolo_shards, cams)
                    yolo_cpus = str(args.yolo_worker_cpus or '')
                    for sidx, shard_cams in enumerate(shards):
                        shard_id = _safe_worker_id_for_cams(shard_cams)
                        setattr(args, '_current_yolo_shard_index', int(sidx))
                        shard_cfg = _write_yolo_shard_config(args.mvs_config, project_root, shard_id, shard_cams, args)
                        _omp = str(max(1, int(getattr(args, 'yolo_worker_omp_num_threads', 1) or 1)))
                        extra_env = {
                            'YOLO_WORKER_ID': shard_id,
                            'YOLO_WORKER_CAMS': ','.join(shard_cams),
                            'OMP_NUM_THREADS': _omp,
                            'MKL_NUM_THREADS': _omp,
                            'OPENBLAS_NUM_THREADS': _omp,
                            'NUMEXPR_NUM_THREADS': _omp,
                        }
                        yolo_cmd = _env_limited_python_cmd([py, mvs_script, '--config', str(shard_cfg)], extra_env=extra_env, cpu_threads=int(args.yolo_worker_omp_num_threads or 1))
                        cpu = _split_worker_cpu(yolo_cpus, sidx)
                        yolo_cmd = _taskset_cmd(cpu, yolo_cmd)
                        p = make_proc(_display_name_for_yolo_worker(shard_id), yolo_cmd, critical=True)
                        procs.append(p); p.start()
                elif run_legacy_yolo:
                    p = make_proc('MVS_AGG', _env_limited_python_cmd([py, mvs_script, '--config', args.mvs_config], cpu_threads=int(args.yolo_worker_omp_num_threads or 1)), critical=True)
                    procs.append(p); p.start()
            else:
                p = make_proc('MVS', _env_limited_python_cmd([py, mvs_script, '--config', args.mvs_config], cpu_threads=int(args.yolo_worker_omp_num_threads or 1)), critical=True)
                procs.append(p); p.start()
                start_global_yolo_if_enabled()
                _write_yolo_chain_mode_status(
                    project_root, args, cams,
                    legacy_enabled=True,
                    reason="enabled_by_monolithic_mvs_path",
                )

        if args.global_person_id_enable:
            gpid_cmd = [
                py, global_person_id_script,
                '--cams', cam_arg,
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--human-jsonl-template', str(args.global_person_id_human_jsonl_template),
                '--output-jsonl', str(args.global_person_id_output_jsonl),
                '--latest-json', str(args.global_person_id_latest_json),
                '--status-json', str(args.global_person_id_status_json),
                '--control-json', str(args.global_person_id_control_json),
                '--control-poll-ms', str(args.global_person_id_control_poll_ms),
                '--stream-role', 'official',
                '--poll-ms', str(args.global_person_id_poll_ms),
                '--sync-delay', str(args.global_person_id_sync_delay),
                '--coord-transform-mode', str(args.global_person_id_coord_transform_mode),
                '--field-width-m', str(args.global_bev_field_width_m),
                '--field-height-m', str(args.global_bev_field_height_m),
                '--field-origin-x-m', str(args.global_bev_field_origin_x_m),
                '--field-origin-y-m', str(args.global_bev_field_origin_y_m),
                '--field-margin-m', str(args.global_person_id_field_margin_m),
                '--max-match-speed-mps', str(args.global_person_id_max_match_speed_mps),
                '--max-update-speed-mps', str(args.global_person_id_max_update_speed_mps),
                '--publish-lost-max-age-ms', str(args.global_person_id_publish_lost_max_age_ms),
                '--publish-min-hits', str(args.global_person_id_publish_min_hits),
                '--cluster-gate-m', str(args.global_person_id_cluster_gate_m),
                '--same-cam-cluster-gate-m', str(args.global_person_id_same_cam_cluster_gate_m),
                '--cross-camera-cluster-gate-m', str(args.global_person_id_cross_camera_cluster_gate_m),
                '--handoff-match-gate-m', str(args.global_person_id_handoff_match_gate_m),
                '--handoff-birth-near-track-suppress-m', str(args.global_person_id_handoff_birth_near_track_suppress_m),
                '--handoff-duplicate-suppress-dist-m', str(args.global_person_id_handoff_duplicate_suppress_dist_m),
                '--handoff-strict-birth-block-m', str(args.global_person_id_handoff_strict_birth_block_m),
                '--handoff-strict-publish-hold-ms', str(args.global_person_id_handoff_strict_publish_hold_ms),
                '--count-gate-x-min-m', str(args.global_person_id_count_gate_x_min_m),
                '--count-gate-x-max-m', str(args.global_person_id_count_gate_x_max_m),
                '--count-gate-y-min-m', str(args.global_person_id_count_gate_y_min_m),
                '--count-gate-y-max-m', str(args.global_person_id_count_gate_y_max_m),
                '--count-gate-short-edge-band-m', str(args.global_person_id_count_gate_short_edge_band_m),
                '--count-gate-short-edge-y-pad-m', str(args.global_person_id_count_gate_short_edge_y_pad_m),
                '--count-gate-startup-bootstrap-ms', str(args.global_person_id_count_gate_startup_bootstrap_ms),
                '--count-gate-bootstrap-allow-anywhere-until-track-count', str(args.global_person_id_count_gate_bootstrap_allow_anywhere_until_track_count),
                '--count-gate-non-short-edge-lost-hold-ms', str(args.global_person_id_count_gate_non_short_edge_lost_hold_ms),
                '--count-gate-non-short-edge-retire-hard-cap-ms', str(args.global_person_id_count_gate_non_short_edge_retire_hard_cap_ms),
                '--duplicate-suppress-dist-m', str(args.global_person_id_duplicate_suppress_dist_m),
                '--match-gate-m', str(args.global_person_id_match_gate_m),
                '--lost-match-gate-m', str(args.global_person_id_lost_match_gate_m),
                '--match-score-min', str(args.global_person_id_match_score_min),
                '--birth-confirm-hits', str(args.global_person_id_birth_confirm_hits),
                '--birth-match-gate-m', str(args.global_person_id_birth_match_gate_m),
                '--birth-near-track-suppress-m', str(args.global_person_id_birth_near_track_suppress_m),
                '--birth-candidate-ttl-ms', str(args.global_person_id_birth_candidate_ttl_ms),
                '--reid-memory-ms', str(args.global_person_id_reid_memory_ms),
                '--retire-after-ms', str(args.global_person_id_retire_after_ms),
                '--filter-backend', str(args.global_person_id_filter_backend),
                '--position-alpha', str(args.global_person_id_position_alpha),
                '--velocity-alpha', str(args.global_person_id_velocity_alpha),
                '--kalman-process-noise-m', str(args.global_person_id_kalman_process_noise_m),
                '--kalman-measurement-noise-m', str(args.global_person_id_kalman_measurement_noise_m),
                '--kalman-initial-pos-var', str(args.global_person_id_kalman_initial_pos_var),
                '--kalman-initial-vel-var', str(args.global_person_id_kalman_initial_vel_var),
                '--kalman-max-innovation-m', str(args.global_person_id_kalman_max_innovation_m),
            ]
            if args.global_person_id_drop_out_of_field:
                gpid_cmd += ['--drop-out-of-field']
            if args.global_person_id_duplicate_suppress_enable:
                gpid_cmd += ['--duplicate-suppress-enable']
            else:
                gpid_cmd += ['--no-duplicate-suppress-enable']
            if args.global_person_id_birth_quarantine_enable:
                gpid_cmd += ['--birth-quarantine-enable']
            else:
                gpid_cmd += ['--no-birth-quarantine-enable']
            if args.global_person_id_handoff_aware_enable:
                gpid_cmd += ['--handoff-aware-enable']
            else:
                gpid_cmd += ['--no-handoff-aware-enable']
            if args.global_person_id_handoff_strict_count_lock_enable:
                gpid_cmd += ['--handoff-strict-count-lock-enable']
            else:
                gpid_cmd += ['--no-handoff-strict-count-lock-enable']
            if args.global_person_id_count_gate_enable:
                gpid_cmd += ['--count-gate-enable']
            else:
                gpid_cmd += ['--no-count-gate-enable']
            if args.global_person_id_kalman_use_for_association:
                gpid_cmd += ['--kalman-use-for-association']
            else:
                gpid_cmd += ['--no-kalman-use-for-association']
            if args.global_person_id_kalman_reject_innovation_enable:
                gpid_cmd += ['--kalman-reject-innovation-enable']
            else:
                gpid_cmd += ['--no-kalman-reject-innovation-enable']
            gpid_cmd = _taskset_cmd(str(args.global_person_id_cpus or ''), gpid_cmd)
            p = make_proc('GLOBAL_PID', gpid_cmd, critical=False)
            procs.append(p); p.start()
            if bool(getattr(args, 'global_person_id_shadow_enable', False)):
                gpid_shadow_cmd = [
                    py, global_person_id_script,
                    '--cams', cam_arg,
                    '--sync-root', str((project_root / 'sync_ipc').resolve()),
                    '--human-jsonl-template', str(args.global_person_id_human_jsonl_template),
                    '--output-jsonl', str(args.global_person_id_shadow_output_jsonl),
                    '--latest-json', str(args.global_person_id_shadow_latest_json),
                    '--status-json', str(args.global_person_id_shadow_status_json),
                    '--control-json', str(args.global_person_id_shadow_control_json),
                    '--control-poll-ms', str(args.global_person_id_control_poll_ms),
                    '--stream-role', 'shadow',
                    '--poll-ms', str(args.global_person_id_poll_ms),
                    '--sync-delay', str(args.global_person_id_sync_delay),
                    '--coord-transform-mode', str(args.global_person_id_coord_transform_mode),
                    '--field-width-m', str(args.global_bev_field_width_m),
                    '--field-height-m', str(args.global_bev_field_height_m),
                    '--field-origin-x-m', str(args.global_bev_field_origin_x_m),
                    '--field-origin-y-m', str(args.global_bev_field_origin_y_m),
                    '--field-margin-m', str(args.global_person_id_field_margin_m),
                    '--max-match-speed-mps', str(args.global_person_id_max_match_speed_mps),
                    '--max-update-speed-mps', str(args.global_person_id_max_update_speed_mps),
                    '--publish-lost-max-age-ms', str(args.global_person_id_publish_lost_max_age_ms),
                    '--publish-min-hits', str(args.global_person_id_publish_min_hits),
                    '--cluster-gate-m', str(args.global_person_id_cluster_gate_m),
                    '--same-cam-cluster-gate-m', str(args.global_person_id_same_cam_cluster_gate_m),
                    '--cross-camera-cluster-gate-m', str(args.global_person_id_cross_camera_cluster_gate_m),
                    '--handoff-match-gate-m', str(args.global_person_id_handoff_match_gate_m),
                    '--handoff-birth-near-track-suppress-m', str(args.global_person_id_handoff_birth_near_track_suppress_m),
                    '--handoff-duplicate-suppress-dist-m', str(args.global_person_id_handoff_duplicate_suppress_dist_m),
                    '--handoff-strict-birth-block-m', str(args.global_person_id_handoff_strict_birth_block_m),
                    '--handoff-strict-publish-hold-ms', str(args.global_person_id_handoff_strict_publish_hold_ms),
                    '--count-gate-x-min-m', str(args.global_person_id_count_gate_x_min_m),
                    '--count-gate-x-max-m', str(args.global_person_id_count_gate_x_max_m),
                    '--count-gate-y-min-m', str(args.global_person_id_count_gate_y_min_m),
                    '--count-gate-y-max-m', str(args.global_person_id_count_gate_y_max_m),
                    '--count-gate-short-edge-band-m', str(args.global_person_id_count_gate_short_edge_band_m),
                    '--count-gate-short-edge-y-pad-m', str(args.global_person_id_count_gate_short_edge_y_pad_m),
                    '--count-gate-startup-bootstrap-ms', str(args.global_person_id_count_gate_startup_bootstrap_ms),
                    '--count-gate-bootstrap-allow-anywhere-until-track-count', str(args.global_person_id_count_gate_bootstrap_allow_anywhere_until_track_count),
                    '--count-gate-non-short-edge-lost-hold-ms', str(args.global_person_id_count_gate_non_short_edge_lost_hold_ms),
                    '--count-gate-non-short-edge-retire-hard-cap-ms', str(args.global_person_id_count_gate_non_short_edge_retire_hard_cap_ms),
                    '--duplicate-suppress-dist-m', str(args.global_person_id_duplicate_suppress_dist_m),
                    '--match-gate-m', str(args.global_person_id_match_gate_m),
                    '--lost-match-gate-m', str(args.global_person_id_lost_match_gate_m),
                    '--match-score-min', str(args.global_person_id_match_score_min),
                    '--birth-confirm-hits', str(args.global_person_id_birth_confirm_hits),
                    '--birth-match-gate-m', str(args.global_person_id_birth_match_gate_m),
                    '--birth-near-track-suppress-m', str(args.global_person_id_birth_near_track_suppress_m),
                    '--birth-candidate-ttl-ms', str(args.global_person_id_birth_candidate_ttl_ms),
                    '--reid-memory-ms', str(args.global_person_id_reid_memory_ms),
                    '--retire-after-ms', str(args.global_person_id_retire_after_ms),
                    '--filter-backend', str(args.global_person_id_shadow_filter_backend),
                    '--position-alpha', str(args.global_person_id_position_alpha),
                    '--velocity-alpha', str(args.global_person_id_velocity_alpha),
                    '--kalman-process-noise-m', str(args.global_person_id_kalman_process_noise_m),
                    '--kalman-measurement-noise-m', str(args.global_person_id_kalman_measurement_noise_m),
                    '--kalman-initial-pos-var', str(args.global_person_id_kalman_initial_pos_var),
                    '--kalman-initial-vel-var', str(args.global_person_id_kalman_initial_vel_var),
                    '--kalman-max-innovation-m', str(args.global_person_id_kalman_max_innovation_m),
                ]
                if args.global_person_id_drop_out_of_field:
                    gpid_shadow_cmd += ['--drop-out-of-field']
                if args.global_person_id_duplicate_suppress_enable:
                    gpid_shadow_cmd += ['--duplicate-suppress-enable']
                else:
                    gpid_shadow_cmd += ['--no-duplicate-suppress-enable']
                if args.global_person_id_birth_quarantine_enable:
                    gpid_shadow_cmd += ['--birth-quarantine-enable']
                else:
                    gpid_shadow_cmd += ['--no-birth-quarantine-enable']
                if args.global_person_id_handoff_aware_enable:
                    gpid_shadow_cmd += ['--handoff-aware-enable']
                else:
                    gpid_shadow_cmd += ['--no-handoff-aware-enable']
                if args.global_person_id_handoff_strict_count_lock_enable:
                    gpid_shadow_cmd += ['--handoff-strict-count-lock-enable']
                else:
                    gpid_shadow_cmd += ['--no-handoff-strict-count-lock-enable']
                if args.global_person_id_count_gate_enable:
                    gpid_shadow_cmd += ['--count-gate-enable']
                else:
                    gpid_shadow_cmd += ['--no-count-gate-enable']
                if args.global_person_id_kalman_use_for_association:
                    gpid_shadow_cmd += ['--kalman-use-for-association']
                else:
                    gpid_shadow_cmd += ['--no-kalman-use-for-association']
                if args.global_person_id_kalman_reject_innovation_enable:
                    gpid_shadow_cmd += ['--kalman-reject-innovation-enable']
                else:
                    gpid_shadow_cmd += ['--no-kalman-reject-innovation-enable']
                gpid_shadow_cmd = _taskset_cmd(str(args.global_person_id_shadow_cpus or ''), gpid_shadow_cmd)
                p = make_proc('GLOBAL_PID_SHADOW', gpid_shadow_cmd, critical=False)
                procs.append(p); p.start()
        if bool(getattr(args, 'person_id_dataset_recorder_enable', False)):
            pid_ds_cmd = [
                py, person_id_dataset_recorder_script,
                '--project-root', str(project_root),
                '--cams', cam_arg,
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--human-jsonl-template', str(args.global_person_id_human_jsonl_template),
                '--global-person-jsonl', str(args.global_person_id_output_jsonl),
                '--global-person-latest-json', str(args.global_person_id_latest_json),
                '--global-person-status-json', str(args.global_person_id_status_json),
                '--global-yolo-status-json', str(args.global_yolo_status_json),
                '--system-status-json', 'system_status_latest.json',
                '--operator-annotations-jsonl', 'operator_annotations.jsonl',
                '--control-json', str(args.person_id_dataset_control_json),
                '--status-json', str(args.person_id_dataset_status_json),
                '--latest-json', str(args.person_id_dataset_latest_json),
                '--output-root', str(args.person_id_dataset_output_root),
                '--evidence-output-root', str(args.person_id_dataset_evidence_output_root),
                '--poll-ms', str(args.person_id_dataset_poll_ms),
                '--status-interval-ms', str(args.person_id_dataset_status_interval_ms),
                '--sample-status-interval-ms', str(args.person_id_dataset_sample_status_interval_ms),
                '--zero-person-keep-ms', str(args.person_id_dataset_zero_person_keep_ms),
            ]
            pid_ds_cmd = _taskset_cmd(str(args.person_id_dataset_recorder_cpus or ''), pid_ds_cmd)
            p = make_proc('PERSON_ID_DATASET', pid_ds_cmd, critical=False)
            procs.append(p); p.start()
        if bool(getattr(args, 'event_mask_evidence_enable', False)):
            evidence_cmd = [
                py, event_mask_evidence_script,
                '--cams', cam_arg,
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--human-jsonl-template', '{root}/human_result_{camera}.jsonl',
                '--evidence-json-template', str(args.event_mask_evidence_json_template),
                '--status-json', str(args.event_mask_evidence_status_json),
                '--control-json', str(args.event_mask_evidence_control_json),
                '--period-ms', str(args.event_mask_evidence_period_ms),
                '--stale-ms', str(args.event_mask_evidence_stale_ms),
                '--tail-bytes', str(args.event_mask_evidence_tail_bytes),
            ]
            if bool(getattr(args, 'event_mask_raster_enable', False)):
                evidence_cmd += [
                    '--mask-raster-enable',
                    '--mask-raster-template', str(args.event_mask_raster_template),
                    '--mask-raster-width', str(args.event_mask_raster_width),
                    '--mask-raster-height', str(args.event_mask_raster_height),
                    '--mask-raster-dilate-px', str(args.event_mask_raster_dilate_px),
                    '--mask-raster-erode-px', str(args.event_mask_raster_erode_px),
                    '--mask-raster-min-conf', str(args.event_mask_raster_min_conf),
                    '--mask-raster-mvs-config', project_abs_path(str(args.global_bev_event_layer_homography_config)),
                ]
            else:
                evidence_cmd += ['--disable-mask-raster']
            evidence_cmd = _taskset_cmd(str(args.event_mask_evidence_cpus or ''), evidence_cmd)
            p = make_proc('EVENT_MASK_EVIDENCE', evidence_cmd, critical=False)
            procs.append(p); p.start()
        if args.hit_delay > 0:
            time.sleep(args.hit_delay)
        if not args.skip_hit:
            hit_cmd = [
                py, hit_script,
                '--config', args.mvs_config,
                '--port', str(args.hit_port),
                '--cams', cam_arg,
                '--control-json', project_abs_path(str(args.hit_control_json)),
            ]
            if args.overlay_ws_enable:
                hit_cmd += ['--overlay-ws-enable', '--overlay-ws-host', str(args.overlay_ws_host), '--overlay-ws-port', str(args.overlay_ws_port)]
            p = make_proc('HIT', hit_cmd, critical=True)
            procs.append(p); p.start()
        if args.global_bullet_reprocess_enable:
            gbr_cmd = [
                py, global_bullet_reprocess_script,
                '--project-root', str(project_root),
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--output-dir', str((project_root / str(args.global_bullet_reprocess_output_dir)).resolve()),
                '--mode', str(args.global_bullet_reprocess_mode),
                '--duration-sec', str(args.global_bullet_reprocess_duration_sec),
                '--interval-sec', str(args.global_bullet_reprocess_interval_sec),
                '--max-scan-lines', str(args.global_bullet_reprocess_max_scan_lines),
            ]
            if str(args.global_bullet_reprocess_calib_manifest or '').strip():
                gbr_cmd += ['--calib-manifest', str(args.global_bullet_reprocess_calib_manifest)]
            if bool(args.global_bullet_reprocess_allow_empty):
                gbr_cmd += ['--allow-empty']
            gbr_cmd = _taskset_cmd(str(args.global_bullet_reprocess_cpus or ''), gbr_cmd)
            p = make_proc('GLOBAL_BULLET_REPROCESS', gbr_cmd, critical=False)
            procs.append(p); p.start()
        if bool(getattr(args, 'global_bullet_fusion_enable', False)):
            gbf_cmd = [
                py, global_bullet_fusion_script,
                '--project-root', str(project_root),
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--mode', str(args.global_bullet_fusion_mode),
                '--input-jsonl', str(args.global_bullet_fusion_input_jsonl),
                '--output-jsonl', str(args.global_bullet_fusion_output_jsonl),
                '--latest-json', str(args.global_bullet_fusion_latest_json),
                '--status-json', str(args.global_bullet_fusion_status_json),
                '--calib-template', str(args.global_bullet_fusion_calib_template),
                '--coord-transform-mode', str(args.global_bullet_fusion_transform_mode),
                '--field-origin-x-m', str(args.global_bev_field_origin_x_m),
                '--field-origin-y-m', str(args.global_bev_field_origin_y_m),
                '--field-width-m', str(args.global_bev_field_width_m),
                '--field-height-m', str(args.global_bev_field_height_m),
                '--interval-sec', str(args.global_bullet_fusion_interval_sec),
                '--min-points', str(args.global_bullet_fusion_min_points),
                '--track-ttl-ms', str(args.global_bullet_fusion_track_ttl_ms),
            ]
            gbf_cmd = _taskset_cmd(str(args.global_bullet_fusion_cpus or ''), gbf_cmd)
            p = make_proc('GLOBAL_BULLET_FUSION', gbf_cmd, critical=False)
            procs.append(p); p.start()
        if bool(getattr(args, 'damage_attribution_enable', False)):
            dmg_cmd = [
                py, damage_attribution_script,
                '--project-root', str(project_root),
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--mode', str(args.damage_attribution_mode),
                '--track-jsonl', str(args.global_bullet_fusion_output_jsonl),
                '--track-latest-json', str(args.global_bullet_fusion_latest_json),
                '--hit-jsonl', str(args.damage_attribution_hit_jsonl),
                '--person-latest-json', str(args.global_person_id_latest_json),
                '--events-jsonl', str(args.damage_attribution_events_jsonl),
                '--latest-json', str(args.damage_attribution_latest_json),
                '--status-json', str(args.damage_attribution_status_json),
                '--attribution-source', str(args.damage_attribution_source),
                '--target-id-source', str(args.damage_attribution_target_id_source),
                '--global-bev-status-json', str(args.damage_attribution_global_bev_status_json),
                '--global-bev-display-max-age-ms', str(args.damage_attribution_global_bev_display_max_age_ms),
                '--default-source-global-id', str(args.damage_attribution_default_source_global_id),
                '--score-threshold', str(args.damage_attribution_score_threshold),
                '--interval-sec', str(args.damage_attribution_interval_sec),
                '--person-max-age-ms', str(args.damage_attribution_person_max_age_ms),
                '--person-history-max-snapshots', str(args.damage_attribution_person_history_max_snapshots),
                '--identity-evidence-max-sync-delta', str(args.damage_attribution_identity_evidence_max_sync_delta),
                '--identity-evidence-history-max-age-ms', str(args.damage_attribution_identity_evidence_history_max_age_ms),
                '--identity-spatial-fallback-max-dist-m', str(args.damage_attribution_identity_spatial_fallback_max_dist_m),
                '--identity-spatial-fallback-min-gap-m', str(args.damage_attribution_identity_spatial_fallback_min_gap_m),
                '--identity-spatial-fallback-max-sync-delta', str(args.damage_attribution_identity_spatial_fallback_max_sync_delta),
                '--display-history-max-snapshots', str(args.damage_attribution_display_history_max_snapshots),
                '--display-history-max-age-ms', str(args.damage_attribution_display_history_max_age_ms),
                '--reserved-virtual-global-id', str(args.damage_attribution_reserved_virtual_global_id),
            ]
            if bool(getattr(args, 'damage_attribution_identity_spatial_fallback_enable', True)):
                dmg_cmd += ['--identity-spatial-fallback-enable']
            else:
                dmg_cmd += ['--no-identity-spatial-fallback-enable']
            if bool(getattr(args, 'damage_attribution_identity_spatial_fallback_require_camera', True)):
                dmg_cmd += ['--identity-spatial-fallback-require-camera']
            else:
                dmg_cmd += ['--no-identity-spatial-fallback-require-camera']
            if bool(getattr(args, 'damage_attribution_allow_stable_id_global_fallback', False)):
                dmg_cmd += ['--allow-stable-id-global-fallback']
            dmg_cmd = _taskset_cmd(str(args.damage_attribution_cpus or ''), dmg_cmd)
            p = make_proc('DAMAGE_ATTRIBUTION', dmg_cmd, critical=False)
            procs.append(p); p.start()
        if bool(getattr(args, 'manual_hit_service_enable', False)):
            manual_hit_cmd = [
                py, manual_hit_script,
                '--project-root', str(project_root),
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--mode', str(args.manual_hit_mode),
                '--command-jsonl', str(args.manual_hit_command_jsonl),
                '--confirmed-jsonl', str(args.manual_hit_confirmed_jsonl),
                '--events-jsonl', str(args.manual_hit_events_jsonl),
                '--latest-json', str(args.manual_hit_latest_json),
                '--status-json', str(args.manual_hit_status_json),
                '--control-json', str(args.manual_hit_control_json),
                '--global-bev-status-json', str(args.game_event_bridge_global_bev_status_json),
                '--person-latest-json', str(args.global_person_id_latest_json),
                '--source-global-id', str(args.manual_hit_source_global_id),
                '--interval-sec', str(args.manual_hit_interval_sec),
                '--status-interval-sec', str(args.manual_hit_status_interval_sec),
                '--dedup-ms', str(args.manual_hit_dedup_ms),
                '--target-max-age-ms', str(args.manual_hit_target_max_age_ms),
                '--target-match-max-dist-m', str(args.manual_hit_target_match_max_dist_m),
            ]
            manual_hit_cmd = _taskset_cmd(str(args.manual_hit_cpus or ''), manual_hit_cmd)
            p = make_proc('MANUAL_HIT', manual_hit_cmd, critical=False)
            procs.append(p); p.start()
        if bool(getattr(args, 'foam_dart_speed_enable', False)):
            speed_cmd = [
                py, foam_dart_speed_script,
                '--project-root', str(project_root),
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--mode', str(args.foam_dart_speed_mode),
                '--track-jsonl', str(args.global_bullet_fusion_output_jsonl),
                '--speed-limit-mps', str(args.foam_dart_speed_limit_mps),
                '--warn-speed-mps', str(args.foam_dart_speed_warn_mps),
                '--interval-sec', str(args.foam_dart_speed_interval_sec),
            ]
            speed_cmd = _taskset_cmd(str(args.foam_dart_speed_cpus or ''), speed_cmd)
            p = make_proc('FOAM_DART_SPEED', speed_cmd, critical=False)
            procs.append(p); p.start()
        if bool(getattr(args, 'game_event_bridge_enable', False)):
            bridge_cmd = [
                py, game_event_bridge_script,
                '--project-root', str(project_root),
                '--sync-root', str((project_root / 'sync_ipc').resolve()),
                '--mode', str(args.game_event_bridge_mode),
                '--interval-sec', str(args.game_event_bridge_interval_sec),
                '--id-source', str(args.game_event_bridge_id_source),
                '--global-bev-status-json', str(args.game_event_bridge_global_bev_status_json),
                '--global-bev-display-max-age-ms', str(args.game_event_bridge_global_bev_display_max_age_ms),
                '--person-latest-json', str(args.global_person_id_latest_json),
                '--events-jsonl', str(args.game_event_bridge_events_jsonl),
                '--control-json', str(args.game_event_bridge_control_json),
                '--hit-source-mode', str(args.game_event_bridge_hit_source_mode),
                '--latest-json', str(args.game_event_bridge_latest_json),
                '--status-json', str(args.game_event_bridge_status_json),
                '--sent-jsonl', str(args.game_event_bridge_sent_jsonl),
                '--windows-host', str(args.game_event_bridge_windows_host),
                '--windows-port', str(args.game_event_bridge_windows_port),
                '--presence-hz', str(args.game_event_bridge_presence_hz),
                '--presence-max-age-ms', str(args.game_event_bridge_presence_max_age_ms),
                '--reconnect-sec', str(args.game_event_bridge_reconnect_sec),
                '--force-hit-source-global-id', str(args.game_event_bridge_force_hit_source_global_id),
                '--virtual-presence-global-id', str(args.game_event_bridge_virtual_presence_global_id),
                '--virtual-presence-player-id', str(args.game_event_bridge_virtual_presence_player_id),
            ]
            if bool(getattr(args, 'game_event_bridge_send_presence_in_shadow', False)):
                bridge_cmd += ['--send-presence-in-shadow']
            bridge_cmd = _taskset_cmd(str(args.game_event_bridge_cpus or ''), bridge_cmd)
            p = make_proc('GAME_EVENT_BRIDGE', bridge_cmd, critical=False)
            procs.append(p); p.start()
        if args.gl_delay > 0:
            time.sleep(args.gl_delay)
        if not args.skip_gl:
            # GL is preview-only. Closing the window should not necessarily kill
            # camera capture / hit judge, but EVENT/MVS/HIT remain critical.
            if args.preview_backend == 'pyopengl-cuda':
                gl_cmd = [
                    py, cuda_gl_script,
                    '--config', args.mvs_config,
                    '--cams', cam_arg,
                    '--raw-template', str(args.preview_raw_template),
                    '--trajectory-shm', str(args.preview_trajectory_shm),
                    '--video-fps', str(args.preview_video_fps),
                    '--render-fps', str(args.preview_render_fps),
                    '--trajectory-fps', str(args.preview_trajectory_fps),
                    '--trajectory-life-ms', str(args.preview_trajectory_life_ms),
                    '--max-trajectory-segments', str(max(1024, int(args.preview_trajectory_max_segments or 8192))),
                    '--bullet-status-ttl-ms', str(args.preview_bullet_status_ttl_ms),
                    '--cuda-device', str(args.preview_cuda_device),
                    '--tile-width', str(args.preview_tile_width),
                    '--tile-height', str(args.preview_tile_height),
                    '--layout', str(args.preview_layout),
                    '--window-width', str(args.preview_window_width),
                    '--window-height', str(args.preview_window_height),
                    '--vsync', str(args.preview_vsync),
                    '--sync-root', str((project_root / 'sync_ipc').resolve()),
                    '--hit-control-json', project_abs_path(str(args.hit_control_json)),
                    '--global-bev-control-json', project_abs_path(str(args.global_bev_control_json)),
                    '--event-overlay-overview-control-json', project_abs_path(str(args.event_overlay_overview_control_json)),
                    '--global-bev-event-layer-alpha', str(args.global_bev_event_layer_alpha),
                    '--global-bev-event-layer-color-mode', str(args.global_bev_event_layer_color_mode),
                    '--global-bev-event-layer-dilate-px', str(args.global_bev_event_layer_dilate_px),
                    '--preview-control-json', project_abs_path('sync_ipc/preview_control.json'),
                    '--manual-hit-command-json', str(args.manual_hit_command_jsonl),
                    '--manual-hit-control-json', sync_abs_path(str(args.manual_hit_control_json)),
                    '--game-event-bridge-control-json', sync_abs_path(str(args.game_event_bridge_control_json)),
                    '--person-id-dataset-control-json', project_abs_path(str(args.person_id_dataset_control_json)),
                    '--person-id-dataset-status-json', project_abs_path(str(args.person_id_dataset_status_json)),
                    '--person-id-dataset-latest-json', project_abs_path(str(args.person_id_dataset_latest_json)),
                    '--hit-jsonl-template', 'hit_candidate_{camera}.jsonl',
                    '--hit-all-jsonl', 'hit_candidate_all.jsonl',
                    '--human-jsonl-template', 'human_result_{camera}.jsonl',
                    '--human-mask-render-mode', str(args.preview_human_mask_render_mode),
                    '--human-mask-alpha', str(args.preview_human_mask_alpha),
                    '--human-mask-life-ms', str(args.preview_human_mask_life_ms),
                    '--human-label-mode', str(args.preview_human_label_mode),
                    '--overlay-ws-url', f"ws://127.0.0.1:{int(args.overlay_ws_port)}",
                ]

                if args.event_overlay_preview_enable:
                    gl_cmd += ['--event-overlay-preview-enable']
                if args.event_overlay_main_inline_enable:
                    gl_cmd += ['--event-overlay-main-inline-enable']
                event_overlay_standalone = bool(args.event_overlay_overview_process_enable and args.event_overlay_preview_enable and not args.event_overlay_window_disable)
                if args.event_overlay_window_disable or event_overlay_standalone:
                    gl_cmd += ['--event-overlay-window-disable']
                gl_cmd += [
                    '--event-overlay-window-width', str(args.event_overlay_window_width),
                    '--event-overlay-window-height', str(args.event_overlay_window_height),
                    '--event-overlay-window-x', str(args.event_overlay_window_x),
                    '--event-overlay-window-y', str(args.event_overlay_window_y),
                    '--event-overlay-window-fps', str(args.event_overlay_window_fps),
                    '--event-overlay-window-label-fps', str(args.event_overlay_window_label_fps),
                    '--event-overlay-window-font-px', str(args.event_overlay_window_font_px),
                    '--event-overlay-window-layout', str(args.event_overlay_window_layout),
                    '--event-overlay-preview-template', str(args.event_overlay_preview_template),
                    '--event-overlay-sidecar-template', str(args.event_overlay_sidecar_template),
                    '--event-overlay-preview-alpha', str(args.event_overlay_preview_alpha),
                    '--event-overlay-preview-gain', str(args.event_overlay_preview_gain),
                    '--event-overlay-preview-gamma', str(args.event_overlay_preview_gamma),
                    '--event-overlay-preview-min-alpha', str(args.event_overlay_preview_min_alpha),
                    '--event-overlay-preview-dilate-px', str(args.event_overlay_preview_dilate_px),
                    '--event-overlay-preview-dilate-max-pixels', str(args.event_overlay_preview_dilate_max_pixels),
                    '--event-overlay-preview-dilate-backend', str(args.event_overlay_preview_dilate_backend),
                    '--event-overlay-preview-max-age-ms', str(args.event_overlay_preview_max_age_ms),
                    '--event-overlay-sync-mode', str(args.event_overlay_sync_mode),
                    '--event-overlay-max-sync-delta-ms', str(args.event_overlay_max_sync_delta_ms),
                ]
                if args.event_overlay_hide_stale:
                    gl_cmd += ['--event-overlay-hide-stale']
                if args.event_overlay_show_stats:
                    gl_cmd += ['--event-overlay-show-stats']
                if bool(getattr(args, 'preview_manual_hit_click_enable', False)):
                    gl_cmd += ['--manual-hit-click-enable']
                if args.status_log_enable:
                    gl_cmd += ['--status-log-enable']
                if args.status_window_disable:
                    gl_cmd += ['--status-window-disable']
                if args.status_window_poll_ms is not None:
                    gl_cmd += ['--status-window-poll-ms', str(args.status_window_poll_ms)]
                if str(args.status_log_dir or '').strip():
                    gl_cmd += ['--status-log-dir', str(args.status_log_dir)]
                if args.status_log_interval_ms is not None:
                    gl_cmd += ['--status-log-interval-ms', str(args.status_log_interval_ms)]
                if args.status_log_tabs_jsonl:
                    gl_cmd += ['--status-log-tabs-jsonl']
                if args.status_log_latest_disable:
                    gl_cmd += ['--status-log-latest-disable']
                if args.preview_allow_cpu_upload_fallback:
                    gl_cmd += ['--allow-cpu-upload-fallback']
                if args.preview_allow_non_nvidia_gl:
                    gl_cmd += ['--allow-non-nvidia-gl']
                if args.no_force_raise_windows:
                    gl_cmd += ['--no-force-raise-windows']
                p = make_proc('GL_CUDA', gl_cmd, critical=False)
            else:
                p = make_proc('GL', [py, gl_script, '--config', args.mvs_config, '--cams', cam_arg, '--preview-fps', str(args.preview_fps)], critical=False)
            procs.append(p); p.start()

            if args.preview_backend == 'pyopengl-cuda' and bool(args.event_overlay_overview_process_enable and args.event_overlay_preview_enable and not args.event_overlay_window_disable):
                ovl_status_json = str(args.event_overlay_overview_status_json or 'sync_ipc/event_overlay_overview_status.json')
                ovl_control_json = project_abs_path(str(args.event_overlay_overview_control_json or 'sync_ipc/event_overlay_overview_control.json'))
                ovl_cmd = [
                    py, event_overlay_overview_script,
                    '--cams', cam_arg,
                    '--sync-root', str((project_root / 'sync_ipc').resolve()),
                    '--vsync', str(args.preview_vsync),
                    '--layout', str(args.preview_layout),
                    '--event-overlay-preview-enable',
                    '--event-overlay-window-width', str(args.event_overlay_window_width),
                    '--event-overlay-window-height', str(args.event_overlay_window_height),
                    '--event-overlay-window-x', str(args.event_overlay_window_x),
                    '--event-overlay-window-y', str(args.event_overlay_window_y),
                    '--event-overlay-window-fps', str(args.event_overlay_window_fps),
                    '--event-overlay-window-label-fps', str(args.event_overlay_window_label_fps),
                    '--event-overlay-window-font-px', str(args.event_overlay_window_font_px),
                    '--event-overlay-window-layout', str(args.event_overlay_window_layout),
                    '--event-overlay-preview-template', str(args.event_overlay_preview_template),
                    '--event-overlay-sidecar-template', str(args.event_overlay_sidecar_template),
                    '--event-overlay-preview-alpha', str(args.event_overlay_preview_alpha),
                    '--event-overlay-preview-gain', str(args.event_overlay_preview_gain),
                    '--event-overlay-preview-gamma', str(args.event_overlay_preview_gamma),
                    '--event-overlay-preview-min-alpha', str(args.event_overlay_preview_min_alpha),
                    '--event-overlay-preview-dilate-px', str(args.event_overlay_preview_dilate_px),
                    '--event-overlay-preview-dilate-max-pixels', str(args.event_overlay_preview_dilate_max_pixels),
                    '--event-overlay-preview-dilate-backend', str(args.event_overlay_preview_dilate_backend),
                    '--event-overlay-preview-max-age-ms', str(args.event_overlay_preview_max_age_ms),
                    '--event-overlay-sync-mode', str(args.event_overlay_sync_mode),
                    '--event-overlay-max-sync-delta-ms', str(args.event_overlay_max_sync_delta_ms),
                    '--event-overlay-overview-status-json', ovl_status_json,
                    '--event-overlay-overview-control-json', ovl_control_json,
                    '--event-overlay-overview-status-interval-ms', str(args.event_overlay_overview_status_interval_ms),
                ]
                if args.event_overlay_hide_stale:
                    ovl_cmd += ['--event-overlay-hide-stale']
                if args.event_overlay_show_stats:
                    ovl_cmd += ['--event-overlay-show-stats']
                if args.no_force_raise_windows:
                    ovl_cmd += ['--no-force-raise-windows']
                ovl_cmd = _taskset_cmd(str(args.event_overlay_overview_cpus or ''), ovl_cmd)
                p = make_proc('EVENT_OVERVIEW', ovl_cmd, critical=False)
                procs.append(p); p.start()

        if args.global_bev_enable:
            bev_cmd = [
                py, global_bev_script,
                '--project-root', str(project_root),
                '--cams', str(args.global_bev_cams or cam_arg),
                '--sync-root', project_abs_path('sync_ipc'),
                '--raw-template', str(args.preview_raw_template),
                '--calib-template', str(args.global_bev_calib_template),
                '--sync-mode', str(args.global_bev_sync_mode),
                '--max-mvs-sync-delta-ticks', str(args.global_bev_max_mvs_sync_delta_ticks),
                '--mvs-sync-tolerance-ms', str(args.global_bev_sync_tolerance_ms),
                '--mvs-sync-outlier-policy', str(args.global_bev_mvs_sync_outlier_policy),
                '--mvs-sync-tick-ms-est', str(args.global_bev_mvs_sync_tick_ms_est),
                '--presentation-delay-ms', str(args.global_bev_presentation_delay_ms),
                '--alignment-mode', str(args.global_bev_alignment_mode),
                '--alignment-delay-ms', str(args.global_bev_alignment_delay_ms),
                '--alignment-shadow-interval-ms', str(args.global_bev_alignment_shadow_interval_ms),
                '--alignment-hold-warn-ms', str(args.global_bev_alignment_hold_warn_ms),
                '--field-width-m', str(args.global_bev_field_width_m),
                '--field-height-m', str(args.global_bev_field_height_m),
                '--field-origin-x-m', str(args.global_bev_field_origin_x_m),
                '--field-origin-y-m', str(args.global_bev_field_origin_y_m),
                '--meters-per-pixel', str(args.global_bev_meters_per_pixel),
                '--display-fps', str(args.global_bev_display_fps),
                '--output-fps', str(args.global_bev_output_fps),
                '--perf-timing-window', str(args.global_bev_perf_timing_window),
                '--pbo-count', str(args.global_bev_pbo_count),
                '--fbo-stats-interval-ms', str(args.global_bev_fbo_stats_interval_ms),
                '--fbo-stats-stride', str(args.global_bev_fbo_stats_stride),
                '--coverage-interval-ms', str(args.global_bev_coverage_interval_ms),
                '--turf-stat-fps', str(args.global_bev_turf_stat_fps),
                '--person-overlay-build-fps', str(args.global_bev_person_overlay_build_fps),
                '--overlay-text-fps', str(args.global_bev_overlay_text_fps),
                '--window-width', str(args.global_bev_window_width),
                '--window-height', str(args.global_bev_window_height),
                '--output-shm', str(args.global_bev_output_shm),
                '--sidecar-json', str(args.global_bev_sidecar_json),
                '--status-json', str(args.global_bev_status_json),
                '--control-json', project_abs_path(str(args.global_bev_control_json)),
                '--runtime-state-json', str(args.global_bev_runtime_state_json),
                '--global-person-latest-json', str(args.global_bev_global_person_latest_json),
                '--global-person-overlay-coord-mode', str(args.global_bev_global_person_overlay_coord_mode),
                '--global-person-overlay-max-speed-mps', str(args.global_bev_global_person_overlay_max_speed_mps),
                '--global-person-overlay-lost-max-age-ms', str(args.global_bev_global_person_overlay_lost_max_age_ms),
                '--manual-hit-command-json', sync_abs_path(str(args.manual_hit_command_jsonl)),
                '--manual-hit-click-radius-px', '42',
                '--event-bullet-jsonl', str(args.global_bev_event_bullet_jsonl),
                '--event-bullet-overlay-life-ms', str(args.global_bev_event_bullet_overlay_life_ms),
                '--event-bullet-overlay-max-records', str(args.global_bev_event_bullet_overlay_max_records),
                '--event-bullet-overlay-poll-ms', str(args.global_bev_event_bullet_overlay_poll_ms),
                '--event-bullet-overlay-dilate-px', str(args.global_bev_event_bullet_overlay_dilate_px),
                '--event-bullet-overlay-line-width', str(args.global_bev_event_bullet_overlay_line_width),
                '--event-bullet-overlay-min-confidence', str(args.global_bev_event_bullet_overlay_min_confidence),
                '--event-bullet-overlay-max-segment-gap-ms', str(args.global_bev_event_bullet_overlay_max_segment_gap_ms),
                '--event-bullet-overlay-transform-mode', str(args.global_bev_event_bullet_overlay_transform_mode),
                '--event-layer-template', str(args.global_bev_event_layer_template),
                '--event-layer-sidecar-template', str(args.global_bev_event_layer_sidecar_template),
                '--event-layer-backend', str(args.global_bev_event_layer_backend),
                '--event-layer-sparse-sidecar-template', str(args.global_bev_event_layer_sparse_sidecar_template),
                '--event-layer-sparse-max-points-draw', str(args.global_bev_event_layer_sparse_max_points_draw),
                '--event-layer-sparse-candidate-scan-initial', str(args.global_bev_event_layer_sparse_candidate_scan_initial),
                '--event-layer-sparse-candidate-scan-escalated', str(args.global_bev_event_layer_sparse_candidate_scan_escalated),
                '--event-layer-sparse-candidate-escalate-abs-ms', str(args.global_bev_event_layer_sparse_candidate_escalate_abs_ms),
                '--event-layer-sparse-candidate-escalate-ratio', str(args.global_bev_event_layer_sparse_candidate_escalate_ratio),
                '--event-layer-sparse-shadow-interval-ms', str(args.global_bev_event_layer_sparse_shadow_interval_ms),
                '--event-layer-presentation-worker-mode', str(args.global_bev_event_layer_presentation_worker_mode),
                '--event-layer-presentation-worker-interval-ms', str(args.global_bev_event_layer_presentation_worker_interval_ms),
                '--event-layer-presentation-bundle-max-age-ms', str(args.global_bev_event_layer_presentation_bundle_max_age_ms),
                '--event-layer-presentation-target-tolerance-ticks', str(args.global_bev_event_layer_presentation_target_tolerance_ticks),
                '--event-layer-presentation-bundle-ring-size', str(args.global_bev_event_layer_presentation_bundle_ring_size),
                '--event-layer-presentation-prebuild-lookback-ticks', str(args.global_bev_event_layer_presentation_prebuild_lookback_ticks),
                '--event-layer-presentation-prebuild-lookahead-ticks', str(args.global_bev_event_layer_presentation_prebuild_lookahead_ticks),
                '--event-layer-presentation-prebuild-max-per-cycle', str(args.global_bev_event_layer_presentation_prebuild_max_per_cycle),
                '--event-layer-presentation-bundle-cache-reuse-ms', str(args.global_bev_event_layer_presentation_bundle_cache_reuse_ms),
                '--event-layer-homography-config', str(args.global_bev_event_layer_homography_config),
                '--event-layer-alpha', str(args.global_bev_event_layer_alpha),
                '--event-layer-gain', str(args.global_bev_event_layer_gain),
                '--event-layer-min-value', str(args.global_bev_event_layer_min_value),
                '--event-layer-max-age-ms', str(args.global_bev_event_layer_max_age_ms),
                '--event-layer-sync-mode', str(args.global_bev_event_layer_sync_mode),
                '--event-layer-max-sync-delta-ms', str(args.global_bev_event_layer_max_sync_delta_ms),
                '--event-layer-max-sync-delta-ticks', str(args.global_bev_event_layer_max_sync_delta_ticks),
                '--event-layer-sync-normal-ms', str(args.global_bev_event_layer_sync_normal_ms),
                '--event-layer-sync-warn-ms', str(args.global_bev_event_layer_sync_warn_ms),
                '--event-layer-sync-hard-ms', str(args.global_bev_event_layer_sync_hard_ms),
                '--event-layer-hold-last-good-ms', str(args.global_bev_event_layer_hold_last_good_ms),
                '--event-layer-history-sidecar-template', str(args.global_bev_event_layer_history_sidecar_template),
                '--event-layer-sync-map-ring-template', str(args.global_bev_event_layer_sync_map_ring_template),
                '--event-layer-fade-alpha-scale', str(args.global_bev_event_layer_fade_alpha_scale),
                '--event-layer-color-mode', str(args.global_bev_event_layer_color_mode),
                '--event-layer-dilate-px', str(args.global_bev_event_layer_dilate_px),
                '--event-layer-dilate-max-pixels', str(args.global_bev_event_layer_dilate_max_pixels),
                '--event-layer-auto-degrade-fps-threshold', str(args.global_bev_event_layer_auto_degrade_fps_threshold),
                '--event-layer-auto-degrade-upload-p95-ms', str(args.global_bev_event_layer_auto_degrade_upload_p95_ms),
                '--event-layer-auto-degrade-min-dilate-px', str(args.global_bev_event_layer_auto_degrade_min_dilate_px),
                '--event-layer-display-budget-ms', str(args.global_bev_event_layer_display_budget_ms),
                '--event-layer-display-hold-ms', str(args.global_bev_event_layer_display_hold_ms),
                '--event-layer-accumulate-window-ms', str(args.global_bev_event_layer_accumulate_window_ms),
                '--event-layer-accumulate-max-frames', str(args.global_bev_event_layer_accumulate_max_frames),
                '--event-layer-texture-scale', str(args.global_bev_event_layer_texture_scale),
                '--event-layer-history-poll-interval-ms', str(args.global_bev_event_layer_history_poll_interval_ms),
                '--runtime-state-event-layer-display-restore-enable' if bool(getattr(args, 'global_bev_runtime_state_event_layer_display_restore_enable', False)) else '--no-runtime-state-event-layer-display-restore-enable',
                '--status-debug-interval-ms', str(args.global_bev_status_debug_interval_ms),
                '--debug-mode', str(args.global_bev_debug_mode),
                '--texture-format', str(args.global_bev_texture_format),
                '--gain', str(args.global_bev_gain),
                '--black-level', str(args.global_bev_black_level),
                '--gamma', str(args.global_bev_gamma),
                '--color-gain-r', str(args.global_bev_color_gain_r),
                '--color-gain-g', str(args.global_bev_color_gain_g),
                '--color-gain-b', str(args.global_bev_color_gain_b),
                '--turf-match-strength', str(args.global_bev_turf_match_strength),
                '--turf-target-rgb', str(args.global_bev_turf_target_rgb),
                '--frame-hold-ms', str(args.global_bev_frame_hold_ms),
                '--unstable-copy-retries', str(args.global_bev_unstable_copy_retries),
                '--mvs-upload-budget-ms', str(args.global_bev_mvs_upload_budget_ms),
                '--field-transform-mode', str(args.global_bev_field_transform_mode),
            ]
            if args.global_bev_turf_match_enable:
                bev_cmd += ['--turf-match-enable']
            if args.global_bev_turf_auto_match:
                bev_cmd += ['--turf-auto-match']
            if args.global_bev_global_person_overlay_enable:
                bev_cmd += ['--global-person-overlay-enable']
            if bool(getattr(args, 'preview_manual_hit_click_enable', False)):
                bev_cmd += ['--manual-hit-click-enable']
            if args.global_bev_event_bullet_overlay_enable:
                bev_cmd += ['--event-bullet-overlay-enable']
            else:
                bev_cmd += ['--no-event-bullet-overlay-enable']
            if args.global_bev_event_layer_enable:
                bev_cmd += ['--event-layer-enable']
            else:
                bev_cmd += ['--no-event-layer-enable']
            if bool(getattr(args, 'global_bev_presentation_delay_enable', False)):
                bev_cmd += ['--presentation-delay-enable']
            else:
                bev_cmd += ['--no-presentation-delay-enable']
            if bool(getattr(args, 'global_bev_event_layer_history_enable', False)):
                bev_cmd += ['--event-layer-history-enable']
            else:
                bev_cmd += ['--no-event-layer-history-enable']
            if bool(getattr(args, 'global_bev_event_layer_hold_last_good_on_hard_outlier', True)):
                bev_cmd += ['--event-layer-hold-last-good-on-hard-outlier']
            else:
                bev_cmd += ['--no-event-layer-hold-last-good-on-hard-outlier']
            if bool(getattr(args, 'global_bev_event_layer_sparse_candidate_escalate_enable', True)):
                bev_cmd += ['--event-layer-sparse-candidate-escalate-enable']
            else:
                bev_cmd += ['--no-event-layer-sparse-candidate-escalate-enable']
            if bool(getattr(args, 'global_bev_event_layer_sparse_shadow_enable', False)):
                bev_cmd += ['--event-layer-sparse-shadow-enable']
            else:
                bev_cmd += ['--no-event-layer-sparse-shadow-enable']
            if bool(getattr(args, 'global_bev_event_layer_presentation_worker_enable', False)):
                bev_cmd += ['--event-layer-presentation-worker-enable']
            else:
                bev_cmd += ['--no-event-layer-presentation-worker-enable']
            if bool(getattr(args, 'global_bev_event_layer_dilate_stabilize_enable', True)):
                bev_cmd += ['--event-layer-dilate-stabilize-enable']
            else:
                bev_cmd += ['--no-event-layer-dilate-stabilize-enable']
            if bool(getattr(args, 'global_bev_event_layer_auto_degrade_enable', True)):
                bev_cmd += ['--event-layer-auto-degrade-enable']
            else:
                bev_cmd += ['--no-event-layer-auto-degrade-enable']
            if bool(getattr(args, 'global_bev_event_layer_display_budget_enable', True)):
                bev_cmd += ['--event-layer-display-budget-enable']
            else:
                bev_cmd += ['--no-event-layer-display-budget-enable']
            if bool(getattr(args, 'global_bev_event_layer_accumulate_enable', True)):
                bev_cmd += ['--event-layer-accumulate-enable']
            else:
                bev_cmd += ['--no-event-layer-accumulate-enable']
            if bool(getattr(args, 'global_bev_mvs_upload_budget_enable', True)):
                bev_cmd += ['--mvs-upload-budget-enable']
            else:
                bev_cmd += ['--no-mvs-upload-budget-enable']
            if args.global_bev_event_bullet_overlay_label_enable:
                bev_cmd += ['--event-bullet-overlay-label-enable']
            else:
                bev_cmd += ['--no-event-bullet-overlay-label-enable']
            if args.global_bev_runtime_state_enable:
                bev_cmd += ['--runtime-state-enable']
            else:
                bev_cmd += ['--no-runtime-state-enable']
            if args.global_bev_debug_gray:
                bev_cmd += ['--debug-gray']
            if args.global_bev_texture_y_flip:
                bev_cmd += ['--texture-y-flip']
            if args.global_bev_grid_enable:
                bev_cmd += ['--grid-enable']
            if args.global_bev_field_flip_x:
                bev_cmd += ['--field-flip-x']
            if args.global_bev_field_flip_y:
                bev_cmd += ['--field-flip-y']
            bev_cmd = _taskset_cmd(str(args.global_bev_cpus or ''), bev_cmd)
            p = make_proc('GLOBAL_BEV', bev_cmd, critical=False)
            procs.append(p); p.start()

        if args.auto_cpu_hotspots_enable:
            collect_cmd = _cpu_hotspot_collect_cmd(
                project_root,
                duration_sec=int(args.auto_cpu_hotspots_duration_sec),
                delay_sec=float(args.auto_cpu_hotspots_delay_sec),
                prefix=str(args.auto_cpu_hotspots_prefix),
            )
            p = make_proc('CPU_HOTSPOTS', collect_cmd, critical=False, startup_lines=120)
            procs.append(p); p.start()

        print("[launcher] all requested processes started. Press Ctrl+C to stop all.", flush=True)
        reported_exits = set()
        while not stopping:
            any_running = False
            for p in procs:
                if p.proc is None:
                    continue
                rc = p.proc.poll()
                if rc is None:
                    any_running = True
                    continue
                if p.name not in reported_exits:
                    reported_exits.add(p.name)
                    level = "ERROR" if p.critical else "INFO"
                    log_launcher(f"{p.name} exited rc={rc} critical={p.critical}", level=level)
                if p.critical:
                    log_launcher(f"critical process {p.name} exited; stopping whole stack", level="ERROR")
                    stop_all()
                    return 1
            if not any_running:
                log_launcher('all child processes exited.')
                break
            time.sleep(0.5)
    except Exception as exc:
        log_launcher(f"launcher exception: {type(exc).__name__}: {exc}", level="ERROR")
        try:
            with open(launcher_main_log, "a", encoding="utf-8", buffering=1) as fp:
                fp.write(traceback.format_exc())
                fp.write("\n")
        except Exception:
            pass
        raise
    finally:
        stop_all()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
