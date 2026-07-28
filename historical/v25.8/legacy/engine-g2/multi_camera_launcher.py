#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_camera_launcher.py — 采集机多相机多进程启动器（修正版）

修复点：
1. JSON 中 list/tuple 参数正确展开为多个命令行参数
   例如: "crop_region": [0, 0, 1279, 400]
   会展开成: --crop-region 0 0 1279 400

2. cmd 中所有元素统一转成 str，避免：
   TypeError: can only join an iterable
   / sequence item 里混入 int / float / bool

3. Linux 下用 os.setsid 创建独立进程组，方便 stop_camera 用 killpg 正常结束子进程

4. 打印命令时使用 shlex.join，更清晰
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ARENA_NAME_PREFIX = "一时兴起竞技场"


def _index_to_alias(index: int) -> str:
    """
    把 0,1,2,... 映射成 A,B,C,...,Z,AA,AB,...
    """
    index = int(index)
    if index < 0:
        return "A"
    chars: List[str] = []
    while True:
        index, rem = divmod(index, 26)
        chars.append(chr(ord('A') + rem))
        if index == 0:
            break
        index -= 1
    return ''.join(reversed(chars))


def _assign_camera_aliases(camera_configs: List['CameraConfig']) -> List['CameraConfig']:
    """
    给未显式提供 alias 的相机按顺序补 A/B/C/D...。
    已经在配置文件里写了 alias 的，保持原样不覆盖。
    """
    used_aliases = {
        str(cfg.alias).strip().upper()
        for cfg in camera_configs
        if str(cfg.alias).strip()
    }
    next_index = 0
    for cfg in camera_configs:
        if str(cfg.alias).strip():
            continue
        while _index_to_alias(next_index) in used_aliases:
            next_index += 1
        cfg.alias = _index_to_alias(next_index)
        used_aliases.add(str(cfg.alias).strip().upper())
        next_index += 1
    return camera_configs


def _display_name(alias: str) -> str:
    alias = str(alias).strip()
    return f"{ARENA_NAME_PREFIX}{alias}" if alias else ARENA_NAME_PREFIX


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    serial: str
    alias: str = ""
    enable_tsf1_pub: bool = True
    enable_hit_events: bool = True
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraProcess:
    config: CameraConfig
    process: Optional[subprocess.Popen] = None
    pid: int = 0
    start_time: float = 0.0
    restart_count: int = 0


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on", "enable", "enabled"}


def _int_cfg(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float_cfg(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_project_path(project_root: str, value: Any, default_rel: str) -> str:
    raw = str(value if value not in (None, "") else default_rel).strip()
    path = os.path.expanduser(raw)
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    return os.path.abspath(path)


def _clean_fifo_alias(alias: str) -> str:
    out = []
    for ch in str(alias or "cam"):
        out.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    cleaned = "".join(out).strip("_")
    return cleaned or "cam"


def _first_option(shared: Dict[str, Any], cameras: List[CameraConfig], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in shared and shared[key] not in (None, ""):
            return shared[key]
    for cfg in cameras:
        for key in keys:
            if key in cfg.extra_args and cfg.extra_args[key] not in (None, ""):
                return cfg.extra_args[key]
    return default


# ---------------------------------------------------------------------------
# 自动检测 Metavision 相机
# ---------------------------------------------------------------------------

def detect_metavision_cameras() -> List[str]:
    """
    尝试检测所有连接的 Metavision 事件相机序列号。
    """
    try:
        from metavision_core.event_io.raw_reader import enumerate_devices
        devices = enumerate_devices()
        serials: List[str] = []
        for dev in devices:
            serial = dev.get("serial", None)
            if serial:
                serials.append(str(serial))
        return serials
    except Exception:
        print("[warn] 无法通过 metavision_core 自动检测相机，尝试其他方式...")

    try:
        result = subprocess.run(
            ["metavision_hal_test", "--list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            serials: List[str] = []
            for line in result.stdout.splitlines():
                if "Serial:" in line or "serial" in line.lower():
                    parts = line.split(":")
                    if len(parts) >= 2:
                        serials.append(parts[-1].strip())
            return serials
    except Exception:
        pass

    print("[error] 无法自动检测相机序列号，请手动指定 --serials 或使用 --config")
    return []


# ---------------------------------------------------------------------------
# 参数构建辅助
# ---------------------------------------------------------------------------

def _normalize_arg_name(key: str) -> str:
    """
    把 JSON key 转成命令行参数名。

    兼容两种写法：
    - crop_region  -> --crop-region
    - csvdebug-track -> --csvdebug-track

    额外兼容旧配置里的短参数写法：
    - i -> --input-event-file
    - s -> --process-from
    - e -> --process-to
    - o -> --out-video
    - l -> --log-results
    这样 cameras.json 里原来写 "i": "xxx.raw" 不会被错误展开成主脚本不认识的 --i。
    """
    key_s = str(key).strip()
    short_alias = {
        "i": "--input-event-file",
        "s": "--process-from",
        "e": "--process-to",
        "o": "--out-video",
        "l": "--log-results",
    }
    if key_s in short_alias:
        return short_alias[key_s]
    return f"--{key_s.replace('_', '-')}"


def _append_one_arg(cmd: List[str], key: str, value: Any) -> None:
    """
    把一个 JSON 配置项追加到命令行参数列表中。

    规则：
    - bool True  -> 只加 flag
    - bool False -> 不加
    - list/tuple -> 展开成多个值
    - 其他       -> 单值参数

    特殊规则：
    - set_bias / set-bias 是可重复参数，JSON 中的 list 会展开成
      --set-bias bias_hpf=5 --set-bias bias_diff_on=5 ...
    """
    key_s = str(key).strip()
    arg_name = _normalize_arg_name(key)

    # Metavision bias 参数是可重复参数，不能按普通 list 参数展开。
    # JSON 写法：
    #   "set_bias": ["bias_hpf=5", "bias_diff_on=5"]
    # 正确命令：
    #   --set-bias bias_hpf=5 --set-bias bias_diff_on=5
    if key_s in {"set_bias", "set-bias"}:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                cmd.extend(["--set-bias", str(item)])
        else:
            cmd.extend(["--set-bias", str(value)])
        return

    if isinstance(value, bool):
        if value:
            cmd.append(arg_name)
        return

    if value is None:
        return

    if isinstance(value, (list, tuple)):
        cmd.append(arg_name)
        cmd.extend(str(v) for v in value)
        return

    cmd.extend([arg_name, str(value)])


def _stringify_cmd(cmd: List[Any]) -> List[str]:
    return [str(x) for x in cmd]


# ---------------------------------------------------------------------------
# 构建单个相机的命令行
# ---------------------------------------------------------------------------

def build_camera_cmd(
    config: CameraConfig,
    server_ip: str,
    tsf1_server_port: int,
    hit_server_port: int,
    shared_options: Dict[str, Any],
    script_path: str,
) -> List[str]:
    """
    为单个相机构建完整命令行。
    """
    cmd: List[Any] = [
        sys.executable,
        script_path,
        "--tsf1-camera-id", str(config.serial),
        "--tsf1-server-ip", str(server_ip),
        "--tsf1-server-port", str(tsf1_server_port),
        "--hit-server-ip", str(server_ip),
        "--hit-server-port", str(hit_server_port),
    ]

    if config.alias:
        cmd.extend(["--hit-camera-alias", str(config.alias)])

    pub_alias = str(config.alias).strip() or str(config.serial)

    # 基础功能开关
    if config.enable_tsf1_pub:
        cmd.extend([
            "--enable-tsf1-pub",
            "--tsf1-auto-start-sender",
            "--tsf1-pub-name", f"bullet_ts_latest_{pub_alias}",
        ])

    if config.enable_hit_events:
        cmd.append("--enable-hit-events")

    overlay_pub_enabled = bool(shared_options.get("enable_event_overlay_pub", False) or config.extra_args.get("enable_event_overlay_pub", False))
    if overlay_pub_enabled:
        raw_name = config.extra_args.get("event_overlay_pub_name", shared_options.get("event_overlay_pub_name", f"event_overlay_latest_{pub_alias}"))
        try:
            overlay_name = str(raw_name).format(camera=pub_alias, cam=pub_alias, alias=pub_alias, serial=config.serial)
        except Exception:
            overlay_name = f"event_overlay_latest_{pub_alias}"
        cmd.extend([
            "--enable-event-overlay-pub",
            "--event-overlay-pub-name", overlay_name,
        ])

    # 这些已经显式加过，不再从 shared_options / extra_args 重复加
    skip_keys = {
        "enable_tsf1_pub",
        "tsf1_auto_start_sender",
        "enable_hit_events",
        "tsf1_pub_name",
        "enable_event_overlay_pub",
        "event_overlay_pub_name",
        "tsf1_camera_id",
        "tsf1_server_ip",
        "tsf1_server_port",
        "hit_server_ip",
        "hit_server_port",
        "hit_camera_alias",
        "enabled",
        "event_source_mode",
        "event_file_fifo_broker_bin",
        "event_file_fifo_broker_raw_files",
        "event_replay_raw_files",
        "event_file_fifo_broker_replay_timing",
        "event_file_fifo_broker_allow_missing_ext_trigger",
        "event_file_fifo_broker_loop",
        "mvs_source_mode",
        "mvs_file_frame_broker_dataset_dir",
        "mvs_file_frame_broker_replay_timing",
        "mvs_file_frame_broker_max_frames",
        "mvs_file_frame_broker_loop",
        "mvs_file_frame_broker_report_ms",
        "mvs_file_frame_broker_cpus",
        "event_worker_cpus",
        "event_native_hal_broker_enable",
        "event_native_hal_broker_bin",
        "event_native_hal_broker_auto_build",
        "event_native_hal_broker_build_dir",
        "event_native_hal_broker_build_log",
        "event_native_hal_broker_out_dir",
        "event_native_hal_broker_fifo_path_template",
        "event_native_hal_broker_fifo_open_mode",
        "event_native_hal_broker_duration_sec",
        "event_native_hal_broker_slice_us",
        "event_native_hal_broker_wait_mode",
        "event_native_hal_broker_poll_sleep_us",
        "event_native_hal_broker_report_ms",
        "event_native_hal_broker_open_retries",
        "event_native_hal_broker_open_retry_delay_ms",
        "event_native_hal_broker_stats_jsonl",
        "event_native_hal_broker_stdout_log",
        "event_native_hal_broker_stderr_log",
        "event_native_hal_broker_startup_timeout_sec",
        "event_native_hal_broker_cpus",
        "event_native_hal_broker_raw_record_command_file",
        "event_native_hal_broker_raw_record_dir",
        "event_native_hal_broker_raw_record_prefix",
        "event_native_hal_broker_raw_record_poll_ms",
        "event_native_hal_broker_enable_trigger_in",
        "event_native_hal_broker_enable_erc",
        "event_native_hal_broker_erc_rate",
        "event_native_hal_broker_enable_trail",
        "event_native_hal_broker_trail_type",
        "event_native_hal_broker_trail_threshold_us",
        "event_native_hal_broker_set_bias",
        "event_native_hal_broker_no_bias_range_bypass",
        "event_visual_mask_overlay_enable",
        "event_visual_mask_raster_template",
        "event_visual_mask_overlay_name_template",
        "event_visual_mask_overlay_sidecar_template",
        "event_visual_mask_overlay_history_enable",
        "event_visual_mask_overlay_history_name_template",
        "event_visual_mask_overlay_history_sidecar_template",
        "event_visual_mask_overlay_history_slots",
        "event_visual_mask_overlay_sparse_enable",
        "event_visual_mask_overlay_sparse_name_template",
        "event_visual_mask_overlay_sparse_sidecar_template",
        "event_visual_mask_overlay_sparse_slots",
        "event_visual_mask_overlay_sparse_max_points",
        "event_visual_mask_no_mask_policy",
        "event_visual_mask_slice_us",
        "event_visual_mask_accumulation_us",
        "event_mask_evidence_control_json",
        "event_visual_mask_publish_fps",
        "event_visual_mask_stale_ms",
        "event_sequential_start_ready_gate_enable",
        "event_sequential_start_timeout_sec",
        "event_sequential_start_poll_sec",
        "event_sequential_start_settle_sec",
        "event_sequential_start_continue_on_failure",
    }

    # 共享参数
    for key, value in shared_options.items():
        if key in skip_keys:
            continue
        _append_one_arg(cmd, key, value)

    # 单相机覆盖参数
    for key, value in config.extra_args.items():
        if key in skip_keys:
            continue
        _append_one_arg(cmd, key, value)

    # 统一转成字符串，避免 join / Popen 出现类型问题
    return _stringify_cmd(cmd)


# ---------------------------------------------------------------------------
# 进程管理
# ---------------------------------------------------------------------------

class MultiCameraManager:
    def __init__(self, global_config: Dict[str, Any], camera_configs: List[CameraConfig]):
        self.global_config = global_config
        self.camera_configs = camera_configs
        self.cameras: Dict[str, CameraProcess] = {}
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.script_path = os.path.join(
            self.project_root,
            "metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py",
        )
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"找不到主脚本: {self.script_path}")

        for cfg in camera_configs:
            self.cameras[cfg.serial] = CameraProcess(config=cfg)
        self.broker_process: Optional[subprocess.Popen] = None
        self.broker_pid: int = 0
        self.broker_start_time: float = 0.0
        self.broker_stdout_fp = None
        self.broker_stderr_fp = None
        self._apply_native_hal_fifo_paths()

    def _shared_options(self) -> Dict[str, Any]:
        shared = self.global_config.get("shared_options", {})
        return shared if isinstance(shared, dict) else {}

    def _native_hal_broker_enabled(self) -> bool:
        shared = self._shared_options()
        source_mode = str(shared.get("event_source_mode", "live_hal") or "live_hal").strip().lower()
        if source_mode == "file_replay":
            return True
        return _bool_cfg(shared.get("event_native_hal_broker_enable"), False)

    def _native_hal_fifo_dir(self) -> str:
        return _resolve_project_path(
            self.project_root,
            self._shared_options().get("event_native_hal_broker_out_dir"),
            "sync_ipc/native_hal_fifos",
        )

    def _native_hal_fifo_path(self, cfg: CameraConfig) -> str:
        alias = cfg.alias or cfg.serial
        return os.path.join(self._native_hal_fifo_dir(), f"event_fifo_{_clean_fifo_alias(alias)}.evb")

    def _apply_native_hal_fifo_paths(self) -> None:
        if not self._native_hal_broker_enabled():
            return
        for cfg in self.camera_configs:
            cfg.extra_args["event_native_hal_fifo_path"] = self._native_hal_fifo_path(cfg)

    def _format_camera_template(self, value: Any, cfg: CameraConfig, default_value: str) -> str:
        raw = str(value if value not in (None, "") else default_value)
        alias = str(cfg.alias or cfg.serial)
        try:
            return raw.format(
                root=os.path.join(self.project_root, "sync_ipc"),
                project_root=self.project_root,
                camera=alias,
                cam=alias,
                id=alias,
                alias=alias,
                serial=str(cfg.serial),
            )
        except Exception:
            return raw

    def _event_ready_flag_path(self, cfg: CameraConfig) -> str:
        shared = self._shared_options()
        alias = str(cfg.alias or cfg.serial)
        raw = cfg.extra_args.get("ext_trigger_ready_file")
        if raw in (None, ""):
            raw = shared.get("ready_file_template")
        value = self._format_camera_template(raw, cfg, os.path.join("sync_ipc", f"event_ready_{alias}.flag"))
        return _resolve_project_path(self.project_root, value, os.path.join("sync_ipc", f"event_ready_{alias}.flag"))

    def _event_worker_status_path(self, cfg: CameraConfig) -> str:
        shared = self._shared_options()
        alias = str(cfg.alias or cfg.serial)
        raw = shared.get("event_worker_status_file", cfg.extra_args.get("event_worker_status_file"))
        value = self._format_camera_template(raw, cfg, os.path.join("sync_ipc", f"event_worker_{alias}_status.json"))
        return _resolve_project_path(self.project_root, value, os.path.join("sync_ipc", f"event_worker_{alias}_status.json"))

    @staticmethod
    def _json_read(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                obj = json.load(fp)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _camera_ready_snapshot(self, serial: str, not_before_wall: float) -> Dict[str, Any]:
        cp = self.cameras[serial]
        cfg = cp.config
        alias = str(cfg.alias or serial)
        now = time.time()
        if cp.process is None:
            return {"ready": False, "reason": "process_not_started", "alias": alias}
        rc = cp.process.poll()
        if rc is not None:
            return {"ready": False, "reason": f"process_exited_rc_{rc}", "alias": alias, "rc": rc}

        flag_path = self._event_ready_flag_path(cfg)
        flag_mtime = 0.0
        if os.path.exists(flag_path):
            try:
                flag_mtime = float(os.path.getmtime(flag_path))
            except OSError:
                flag_mtime = 0.0
            if flag_mtime >= float(not_before_wall) - 0.25:
                return {
                    "ready": True,
                    "reason": "ready_flag",
                    "alias": alias,
                    "ready_age_ms": round((now - flag_mtime) * 1000.0, 3),
                    "ready_file": flag_path,
                }

        status_path = self._event_worker_status_path(cfg)
        status_mtime = 0.0
        status = {}
        if os.path.exists(status_path):
            try:
                status_mtime = float(os.path.getmtime(status_path))
            except OSError:
                status_mtime = 0.0
            status = self._json_read(status_path)
            state = str(status.get("state", "") or "").strip().lower()
            last_error = str(status.get("last_error", "") or "").strip()
            if (
                status_mtime >= float(not_before_wall) - 0.25
                and state not in {"", "error", "failed", "stopping", "stopped"}
                and not last_error
            ):
                return {
                    "ready": True,
                    "reason": f"worker_status_{state or 'fresh'}",
                    "alias": alias,
                    "status_age_ms": round((now - status_mtime) * 1000.0, 3),
                    "status_file": status_path,
                }

        return {
            "ready": False,
            "reason": "waiting_ready_flag_or_status",
            "alias": alias,
            "ready_file": flag_path,
            "ready_file_seen": bool(flag_mtime > 0.0),
            "status_file": status_path,
            "status_file_seen": bool(status_mtime > 0.0),
            "status_state": str(status.get("state", "") or ""),
        }

    def wait_camera_ready(self, serial: str, timeout_s: float, poll_s: float = 0.2) -> Dict[str, Any]:
        start_wall = float(self.cameras[serial].start_time or time.time())
        deadline = time.time() + max(0.1, float(timeout_s))
        last: Dict[str, Any] = {"ready": False, "reason": "not_checked"}
        while time.time() < deadline:
            last = self._camera_ready_snapshot(serial, start_wall)
            if bool(last.get("ready")):
                return last
            if str(last.get("reason", "")).startswith("process_exited"):
                return last
            time.sleep(max(0.05, float(poll_s)))
        last = self._camera_ready_snapshot(serial, start_wall)
        last["ready"] = False
        last["reason"] = f"ready_timeout_{float(timeout_s):.1f}s:{last.get('reason', '')}"
        return last

    def _native_hal_broker_bin(self) -> str:
        shared = self._shared_options()
        source_mode = str(shared.get("event_source_mode", "live_hal") or "live_hal").strip().lower()
        if source_mode == "file_replay":
            return _resolve_project_path(
                self.project_root,
                shared.get("event_file_fifo_broker_bin"),
                "native_event_bridge/build/event_file_fifo_broker",
            )
        return _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_bin"),
            "native_event_bridge/build/hal_event_fifo_broker",
        )

    def _ensure_native_hal_broker_built(self) -> bool:
        broker_bin = self._native_hal_broker_bin()
        if os.path.exists(broker_bin) and os.access(broker_bin, os.X_OK):
            return True
        shared = self._shared_options()
        if not _bool_cfg(shared.get("event_native_hal_broker_auto_build"), True):
            print(f"[broker][ERROR] binary not found: {broker_bin}")
            return False
        build_script = os.path.join(self.project_root, "native_event_bridge", "build_native_event_bridge.sh")
        build_dir = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_build_dir"),
            "native_event_bridge/build",
        )
        log_path = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_build_log"),
            "sync_ipc/launcher_logs/event_native_hal_broker_build.log",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(build_script):
            print(f"[broker][ERROR] build script not found: {build_script}")
            return False
        print(f"[broker] build native HAL broker -> {build_dir}")
        with open(log_path, "w", encoding="utf-8") as fp:
            rc = subprocess.run(
                [build_script, build_dir],
                cwd=self.project_root,
                stdout=fp,
                stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0:
            print(f"[broker][ERROR] build failed rc={rc}, log={log_path}")
            return False
        if not os.path.exists(broker_bin):
            print(f"[broker][ERROR] build completed but binary missing: {broker_bin}")
            return False
        return True

    def _native_hal_broker_cmd(self) -> List[str]:
        shared = self._shared_options()
        source_mode = str(shared.get("event_source_mode", "live_hal") or "live_hal").strip().lower()
        serials = [str(cfg.serial).strip() for cfg in self.camera_configs if str(cfg.serial).strip()]
        aliases = [str(cfg.alias or cfg.serial).strip() for cfg in self.camera_configs if str(cfg.serial).strip()]
        stats_jsonl = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_stats_jsonl"),
            "sync_ipc/event_native_hal_broker_stats.jsonl",
        )
        if source_mode == "file_replay":
            raw_files = str(
                shared.get("event_file_fifo_broker_raw_files")
                or shared.get("event_replay_raw_files")
                or ""
            ).strip()
            if not raw_files:
                per_cam = []
                for cfg in self.camera_configs:
                    raw = str(cfg.extra_args.get("event_replay_raw_file", "") or "").strip()
                    if raw:
                        alias = str(cfg.alias or cfg.serial).strip()
                        per_cam.append(f"{alias}={raw}")
                raw_files = ",".join(per_cam)
            if not raw_files:
                raise RuntimeError("event_source_mode=file_replay requires event_file_fifo_broker_raw_files")
            cmd: List[str] = [
                self._native_hal_broker_bin(),
                "--raw-files", raw_files,
                "--aliases", ",".join(aliases),
                "--output-mode", "fifo",
                "--out-dir", self._native_hal_fifo_dir(),
                "--fifo-open-mode", str(shared.get("event_native_hal_broker_fifo_open_mode", "rdwr") or "rdwr"),
                "--slice-us", str(_int_cfg(shared.get("event_native_hal_broker_slice_us"), 4000)),
                "--report-ms", str(_int_cfg(shared.get("event_native_hal_broker_report_ms"), 1000)),
                "--replay-timing", str(shared.get("event_file_fifo_broker_replay_timing", "realtime") or "realtime"),
                "--stats-jsonl", stats_jsonl,
            ]
            if _bool_cfg(shared.get("event_file_fifo_broker_allow_missing_ext_trigger"), False):
                cmd.append("--allow-missing-ext-trigger")
            if _bool_cfg(shared.get("event_file_fifo_broker_loop"), False):
                cmd.append("--loop")
            cpus = str(shared.get("event_native_hal_broker_cpus", "") or "").strip()
            if cpus and sys.platform != "win32":
                cmd = ["taskset", "-c", cpus] + cmd
            return _stringify_cmd(cmd)

        cmd: List[str] = [
            self._native_hal_broker_bin(),
            "--serials", ",".join(serials),
            "--aliases", ",".join(aliases),
            "--output-mode", "fifo",
            "--out-dir", self._native_hal_fifo_dir(),
            "--fifo-open-mode", str(shared.get("event_native_hal_broker_fifo_open_mode", "rdwr") or "rdwr"),
            "--duration-sec", str(_int_cfg(shared.get("event_native_hal_broker_duration_sec"), 0)),
            "--slice-us", str(_int_cfg(shared.get("event_native_hal_broker_slice_us"), 4000)),
            "--wait-mode", str(shared.get("event_native_hal_broker_wait_mode", "poll") or "poll"),
            "--poll-sleep-us", str(_int_cfg(shared.get("event_native_hal_broker_poll_sleep_us"), 100)),
            "--report-ms", str(_int_cfg(shared.get("event_native_hal_broker_report_ms"), 1000)),
            "--open-retries", str(_int_cfg(shared.get("event_native_hal_broker_open_retries"), 1)),
            "--open-retry-delay-ms", str(_int_cfg(shared.get("event_native_hal_broker_open_retry_delay_ms"), 0)),
            "--stats-jsonl", stats_jsonl,
        ]

        raw_record_command_file = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_raw_record_command_file") or shared.get("raw_record_command_file"),
            "sync_ipc/record_control.json",
        )
        raw_record_dir = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_raw_record_dir") or shared.get("raw_record_dir"),
            "raw_records",
        )
        raw_record_prefix = str(
            shared.get("event_native_hal_broker_raw_record_prefix") or shared.get("raw_record_prefix") or "eventraw"
        )
        raw_record_poll_ms = _int_cfg(
            shared.get("event_native_hal_broker_raw_record_poll_ms") or shared.get("raw_record_command_poll_ms"),
            50,
        )
        cmd.extend([
            "--raw-record-command-file", raw_record_command_file,
            "--raw-record-dir", raw_record_dir,
            "--raw-record-prefix", raw_record_prefix,
            "--raw-record-poll-ms", str(raw_record_poll_ms),
        ])

        trigger_enabled = _bool_cfg(
            _first_option(shared, self.camera_configs, ["event_native_hal_broker_enable_trigger_in", "enable_ext_trigger_in"], False),
            False,
        )
        if trigger_enabled or any(_bool_cfg(cfg.extra_args.get("enable_ext_trigger_in"), False) for cfg in self.camera_configs):
            cmd.append("--enable-trigger-in")

        erc_enabled = _bool_cfg(
            _first_option(shared, self.camera_configs, ["event_native_hal_broker_enable_erc", "enable_erc"], False),
            False,
        )
        if erc_enabled or any(_bool_cfg(cfg.extra_args.get("enable_erc"), False) for cfg in self.camera_configs):
            erc_rate = _first_option(
                shared,
                self.camera_configs,
                ["event_native_hal_broker_erc_rate", "erc_cd_event_rate"],
                10000000,
            )
            cmd.extend(["--enable-erc", "--erc-rate", str(_int_cfg(erc_rate, 10000000))])

        trail_enabled = _bool_cfg(
            _first_option(shared, self.camera_configs, ["event_native_hal_broker_enable_trail", "enable_hw_trail"], False),
            False,
        )
        if trail_enabled or any(_bool_cfg(cfg.extra_args.get("enable_hw_trail"), False) for cfg in self.camera_configs):
            trail_type = _first_option(
                shared,
                self.camera_configs,
                ["event_native_hal_broker_trail_type", "hw_trail_type"],
                "stc_keep_trail",
            )
            trail_th = _first_option(
                shared,
                self.camera_configs,
                ["event_native_hal_broker_trail_threshold_us", "hw_trail_threshold_us"],
                10000,
            )
            cmd.extend([
                "--enable-trail",
                "--trail-type", str(trail_type or "stc_keep_trail"),
                "--trail-threshold-us", str(_int_cfg(trail_th, 10000)),
            ])

        biases = _first_option(shared, self.camera_configs, ["event_native_hal_broker_set_bias", "set_bias"], [])
        if isinstance(biases, (list, tuple)):
            for bias in biases:
                if bias not in (None, ""):
                    cmd.extend(["--set-bias", str(bias)])
        elif biases not in (None, ""):
            cmd.extend(["--set-bias", str(biases)])
        if _bool_cfg(shared.get("event_native_hal_broker_no_bias_range_bypass"), False):
            cmd.append("--no-bias-range-bypass")

        if _bool_cfg(shared.get("event_visual_mask_overlay_enable"), False):
            visual_mask_control_json = _resolve_project_path(
                self.project_root,
                shared.get("event_mask_evidence_control_json"),
                "sync_ipc/event_mask_control.json",
            )
            cmd.extend([
                "--visual-mask-overlay-enable",
                "--visual-mask-raster-template", str(shared.get("event_visual_mask_raster_template") or "/dev/shm/event_human_mask_raster_{camera}.mmap"),
                "--visual-mask-overlay-name-template", str(shared.get("event_visual_mask_overlay_name_template") or "event_overlay_masked_latest_{camera}"),
                "--visual-mask-overlay-sidecar-template", str(shared.get("event_visual_mask_overlay_sidecar_template") or "sync_ipc/event_overlay_masked_{camera}_latest.json"),
                "--visual-mask-no-mask-policy", str(shared.get("event_visual_mask_no_mask_policy") or "hold_last_good"),
                "--visual-mask-slice-us", str(_int_cfg(shared.get("event_visual_mask_slice_us"), 33333)),
                "--visual-mask-accumulation-us", str(_int_cfg(shared.get("event_visual_mask_accumulation_us"), 33333)),
                "--visual-mask-publish-fps", str(_int_cfg(shared.get("event_visual_mask_publish_fps"), 40)),
                "--visual-mask-stale-ms", str(_int_cfg(shared.get("event_visual_mask_stale_ms"), 120)),
                "--visual-mask-control-json", visual_mask_control_json,
            ])
            if _bool_cfg(shared.get("event_visual_mask_overlay_history_enable"), False):
                cmd.extend([
                    "--visual-mask-overlay-history-enable",
                    "--visual-mask-overlay-history-name-template", str(shared.get("event_visual_mask_overlay_history_name_template") or "event_overlay_masked_hist_{camera}_{slot}"),
                    "--visual-mask-overlay-history-sidecar-template", str(shared.get("event_visual_mask_overlay_history_sidecar_template") or "sync_ipc/event_overlay_masked_{camera}_history.json"),
                    "--visual-mask-overlay-history-slots", str(_int_cfg(shared.get("event_visual_mask_overlay_history_slots"), 64)),
                ])
            else:
                cmd.append("--disable-visual-mask-overlay-history")
            if _bool_cfg(shared.get("event_visual_mask_overlay_sparse_enable"), False):
                cmd.extend([
                    "--visual-mask-overlay-sparse-enable",
                    "--visual-mask-overlay-sparse-name-template", str(shared.get("event_visual_mask_overlay_sparse_name_template") or "event_overlay_sparse_hist_{camera}_{slot}"),
                    "--visual-mask-overlay-sparse-sidecar-template", str(shared.get("event_visual_mask_overlay_sparse_sidecar_template") or "sync_ipc/event_overlay_sparse_{camera}_history.json"),
                    "--visual-mask-overlay-sparse-slots", str(_int_cfg(shared.get("event_visual_mask_overlay_sparse_slots"), 64)),
                    "--visual-mask-overlay-sparse-max-points", str(_int_cfg(shared.get("event_visual_mask_overlay_sparse_max_points"), 20000)),
                ])
            else:
                cmd.append("--disable-visual-mask-overlay-sparse")
        else:
            cmd.append("--disable-visual-mask-overlay")

        cpus = str(shared.get("event_native_hal_broker_cpus", "") or "").strip()
        if cpus and sys.platform != "win32":
            cmd = ["taskset", "-c", cpus] + cmd
        return _stringify_cmd(cmd)

    def start_native_hal_broker(self) -> bool:
        if not self._native_hal_broker_enabled():
            return True
        if self.broker_process is not None and self.broker_process.poll() is None:
            return True
        if not self._ensure_native_hal_broker_built():
            return False

        fifo_dir = self._native_hal_fifo_dir()
        os.makedirs(fifo_dir, exist_ok=True)
        try:
            cmd = self._native_hal_broker_cmd()
        except Exception as exc:
            print(f"[broker][ERROR] command build failed: {exc}")
            return False
        shared = self._shared_options()
        stdout_log = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_stdout_log"),
            "sync_ipc/launcher_logs/event_native_hal_broker.out",
        )
        stderr_log = _resolve_project_path(
            self.project_root,
            shared.get("event_native_hal_broker_stderr_log"),
            "sync_ipc/launcher_logs/event_native_hal_broker.err",
        )
        os.makedirs(os.path.dirname(stdout_log), exist_ok=True)
        os.makedirs(os.path.dirname(stderr_log), exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MV_HAL_PLUGIN_PATH", "/usr/lib/ids/ueye_evs/hal/plugins")
        self.broker_stdout_fp = open(stdout_log, "w", encoding="utf-8", buffering=1)
        self.broker_stderr_fp = open(stderr_log, "w", encoding="utf-8", buffering=1)
        source_mode = str(shared.get("event_source_mode", "live_hal") or "live_hal").strip().lower()
        print(f"[broker] start event FIFO broker mode={source_mode}: {shlex.join(cmd)}")
        try:
            kwargs: Dict[str, Any] = {}
            if sys.platform != "win32":
                kwargs["preexec_fn"] = os.setsid
            self.broker_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=self.broker_stdout_fp,
                stderr=self.broker_stderr_fp,
                **kwargs,
            )
            self.broker_pid = int(self.broker_process.pid)
            self.broker_start_time = time.time()
        except Exception as exc:
            print(f"[broker][ERROR] start failed: {exc}")
            self._close_broker_logs()
            return False

        timeout_s = max(1.0, _float_cfg(shared.get("event_native_hal_broker_startup_timeout_sec"), 30.0))
        deadline = time.time() + timeout_s
        fifo_paths = [self._native_hal_fifo_path(cfg) for cfg in self.camera_configs]
        while time.time() < deadline:
            if self.broker_process.poll() is not None:
                print(f"[broker][ERROR] exited during startup rc={self.broker_process.returncode}")
                return False
            if all(os.path.exists(p) for p in fifo_paths):
                print(f"[broker] ready pid={self.broker_pid} fifos={len(fifo_paths)} dir={fifo_dir}")
                return True
            time.sleep(0.1)
        print(f"[broker][ERROR] startup timeout waiting for FIFOs in {fifo_dir}")
        return False

    def _close_broker_logs(self) -> None:
        for attr in ("broker_stdout_fp", "broker_stderr_fp"):
            fp = getattr(self, attr, None)
            if fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def stop_native_hal_broker(self) -> None:
        proc = self.broker_process
        if proc is None:
            self._close_broker_logs()
            return
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.broker_pid)], capture_output=True, timeout=5)
                else:
                    try:
                        os.killpg(os.getpgid(self.broker_pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                proc.wait(timeout=5)
                print(f"[broker] stopped pid={self.broker_pid}")
        except subprocess.TimeoutExpired:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.broker_pid)], capture_output=True, timeout=5)
                else:
                    os.killpg(os.getpgid(self.broker_pid), signal.SIGKILL)
                print(f"[broker] killed pid={self.broker_pid}")
            except Exception as exc:
                print(f"[broker][ERROR] kill failed: {exc}")
        finally:
            self.broker_process = None
            self.broker_pid = 0
            self._close_broker_logs()

    def start_camera(self, serial: str) -> bool:
        if serial not in self.cameras:
            print(f"[error] 未知序列号: {serial}")
            return False

        cp = self.cameras[serial]
        if cp.process is not None and cp.process.poll() is None:
            print(f"[warn] 相机 {serial} 已在运行 (PID={cp.pid})")
            return False

        cmd = build_camera_cmd(
            config=cp.config,
            server_ip=self.global_config.get("server_ip", "192.168.1.100"),
            tsf1_server_port=int(self.global_config.get("tsf1_server_port", 5001)),
            hit_server_port=int(self.global_config.get("hit_server_port", 5003)),
            shared_options=self.global_config.get("shared_options", {}),
            script_path=self.script_path,
        )

        alias = cp.config.alias or serial
        print(f"\n[start] 启动相机 {_display_name(alias)} (SN={serial})")
        print(f"  命令: {shlex.join(cmd)}")

        try:
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                script_dir = os.path.dirname(self.script_path)
                env = os.environ.copy()
                env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
                proc = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    env=env,
                    creationflags=creationflags,
                    stdout=None,
                    stderr=None,
                )
            else:
                # Linux / macOS：创建独立进程组，便于 killpg
                script_dir = os.path.dirname(self.script_path)
                env = os.environ.copy()
                env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
                proc = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    env=env,
                    stdout=None,
                    stderr=None,
                    preexec_fn=os.setsid,
                )

            cp.process = proc
            cp.pid = int(proc.pid)
            cp.start_time = time.time()
            cp.restart_count += 1
            print(f"  PID={proc.pid}, 第 {cp.restart_count} 次启动")
            return True

        except Exception as e:
            print(f"[error] 启动相机 {serial} 失败: {e}")
            return False

    def stop_camera(self, serial: str) -> bool:
        if serial not in self.cameras:
            return False

        cp = self.cameras[serial]
        if cp.process is None:
            return True

        alias = cp.config.alias or serial

        try:
            if cp.process.poll() is not None:
                cp.process = None
                cp.pid = 0
                return True

            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(cp.pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                try:
                    os.killpg(os.getpgid(cp.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

            cp.process.wait(timeout=5)
            print(f"[stop] 相机 {alias} (PID={cp.pid}) 已停止")

        except subprocess.TimeoutExpired:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(cp.pid)],
                        capture_output=True,
                        timeout=5,
                    )
                else:
                    os.killpg(os.getpgid(cp.pid), signal.SIGKILL)
                print(f"[stop] 相机 {_display_name(alias)} (PID={cp.pid}) 强制终止")
            except Exception as e:
                print(f"[error] 强制停止相机 {_display_name(alias)} 失败: {e}")

        except Exception as e:
            print(f"[error] 停止相机 {_display_name(alias)} 异常: {e}")

        finally:
            cp.process = None
            cp.pid = 0

        return True

    def restart_camera(self, serial: str) -> bool:
        self.stop_camera(serial)
        time.sleep(1)
        return self.start_camera(serial)

    def start_all(self, max_wait_per_cam: int = 8) -> None:
        success: List[str] = []
        failed: List[str] = []

        print("\n开始依次启动所有相机...\n")

        if self._native_hal_broker_enabled() and not self.start_native_hal_broker():
            print("[broker][ERROR] native HAL FIFO broker startup failed; aborting EVENT startup")
            sys.exit(2)

        serials = list(self.cameras.keys())
        shared = self._shared_options()
        ready_gate_requested = _bool_cfg(shared.get("event_sequential_start_ready_gate_enable"), False)
        ready_gate_enable = bool(ready_gate_requested and not self._native_hal_broker_enabled())
        ready_timeout_s = max(0.1, _float_cfg(shared.get("event_sequential_start_timeout_sec"), float(max_wait_per_cam)))
        ready_poll_s = max(0.05, _float_cfg(shared.get("event_sequential_start_poll_sec"), 0.2))
        settle_s = max(0.0, _float_cfg(shared.get("event_sequential_start_settle_sec"), 0.0))
        continue_on_failure = _bool_cfg(shared.get("event_sequential_start_continue_on_failure"), True)
        if ready_gate_requested and not ready_gate_enable:
            print("[startup] sequential ready gate skipped in native HAL broker mode; broker open/FIFO readiness is authoritative")
        if ready_gate_enable:
            print(
                f"[startup] sequential ready gate enabled: timeout={ready_timeout_s:.1f}s "
                f"poll={ready_poll_s:.2f}s settle={settle_s:.2f}s "
                f"continue_on_failure={continue_on_failure}"
            )
        for i, serial in enumerate(serials):
            ok = self.start_camera(serial)
            if not ok:
                failed.append(serial)
                if ready_gate_enable and not continue_on_failure:
                    break
                continue

            cp = self.cameras[serial]
            alias = cp.config.alias or serial
            print(f"  等待相机 {_display_name(alias)} 初始化 (最多 {max_wait_per_cam}s)...")

            ready_info: Dict[str, Any] = {}
            if ready_gate_enable:
                ready_info = self.wait_camera_ready(serial, ready_timeout_s, ready_poll_s)
            else:
                for _ in range(max_wait_per_cam):
                    time.sleep(1)
                    if cp.process is not None and cp.process.poll() is not None:
                        break

            if cp.process is not None and cp.process.poll() is not None:
                rc = cp.process.returncode
                print(f"  [FAIL] 相机 {_display_name(alias)} (SN={serial}) 启动失败 (exit code={rc})")
                cp.process = None
                cp.pid = 0
                failed.append(serial)
                if ready_gate_enable and not continue_on_failure:
                    break
            elif ready_gate_enable and not bool(ready_info.get("ready")):
                print(
                    f"  [FAIL] camera {_display_name(alias)} (SN={serial}) ready gate timeout: "
                    f"{ready_info.get('reason', 'unknown')}"
                )
                failed.append(serial)
                if not continue_on_failure:
                    break
            else:
                print(f"  [OK] 相机 {_display_name(alias)} (SN={serial}) 启动成功")
                if ready_gate_enable:
                    print(f"  [ready] {_display_name(alias)} (SN={serial}) ready_by={ready_info.get('reason', 'unknown')}")
                success.append(serial)
                if settle_s > 0.0 and i < len(serials) - 1:
                    time.sleep(settle_s)

        print(f"\n{'=' * 50}")
        print(f"启动完成: {len(success)} 台成功, {len(failed)} 台失败")
        for s in success:
            cp = self.cameras[s]
            alias = cp.config.alias or s
            print(f"  ✓ {_display_name(alias)} (SN={s}) PID={cp.pid}")

        if failed:
            for s in failed:
                alias = self.cameras[s].config.alias or s
                print(f"  ✗ {_display_name(alias)} (SN={s})")
        print(f"{'=' * 50}")
        if failed and ready_gate_enable and not continue_on_failure:
            print("[startup][ERROR] sequential ready gate failed; stopping started cameras")
            self.stop_all()
            sys.exit(2)

    def stop_all(self) -> None:
        self.stop_native_hal_broker()
        for serial in list(self.cameras.keys()):
            self.stop_camera(serial)

    def _record_command_file(self) -> str:
        shared = self.global_config.get("shared_options", {})
        return str(shared.get("raw_record_command_file", "/tmp/eventcam_raw_record_cmd.json"))

    def _write_record_command(self, cmd: str, session_id: str | None = None) -> None:
        path = self._record_command_file()
        payload = {
            "seq": int(time.time() * 1000),
            "cmd": str(cmd).lower(),
        }
        if session_id:
            payload["session_id"] = str(session_id)
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
        except OSError as exc:
            print(f"[record][ERROR] write command failed: {path} -> {exc}")
            print("[record][ERROR] 请检查磁盘空间: df -h /tmp /home")
            raise

    def start_record_all(self) -> str:
        session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._write_record_command('start', session_id=session_id)
        print(f"[record] start broadcast -> session={session_id} file={self._record_command_file()}")
        return session_id

    def stop_record_all(self) -> None:
        self._write_record_command('stop')
        print(f"[record] stop broadcast -> file={self._record_command_file()}")

    def status(self) -> str:
        lines = ["\n=== 相机状态 ==="]
        if self._native_hal_broker_enabled():
            broker_running = self.broker_process is not None and self.broker_process.poll() is None
            broker_status = "RUNNING" if broker_running else "STOPPED"
            uptime = ""
            if broker_running:
                uptime = f", running {int(time.time() - self.broker_start_time)}s"
            lines.append(f"  [broker] native HAL FIFO: {broker_status} PID={self.broker_pid or 'N/A'}{uptime}")
        for i, (serial, cp) in enumerate(self.cameras.items()):
            alias = cp.config.alias or serial
            running = cp.process is not None and cp.process.poll() is None
            status = "RUNNING" if running else "STOPPED"
            pid_str = f"PID={cp.pid}" if cp.pid else "N/A"
            uptime = ""
            if running:
                secs = int(time.time() - cp.start_time)
                uptime = f", 运行 {secs}s"
            lines.append(
                f"  [{i}] {_display_name(alias)} (SN={serial}): {status} {pid_str}{uptime}, 重启 {cp.restart_count} 次"
            )
        lines.append("=" * 30)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 交互控制台
# ---------------------------------------------------------------------------

def interactive_console(manager: MultiCameraManager) -> None:
    print("\n多相机管理控制台已启动。输入 'help' 查看可用命令。")
    while True:
        try:
            cmd = input("\n[cmd]> ").strip().split()
            if not cmd:
                continue

            action = cmd[0].lower()

            if action in ("quit", "exit", "q"):
                print("正在停止所有相机...")
                manager.stop_all()
                break

            elif action == "status":
                print(manager.status())

            elif action == "restart" and len(cmd) >= 2:
                idx = int(cmd[1])
                serial = list(manager.cameras.keys())[idx]
                manager.restart_camera(serial)

            elif action == "stop" and len(cmd) >= 2:
                idx = int(cmd[1])
                serial = list(manager.cameras.keys())[idx]
                manager.stop_camera(serial)

            elif action == "start" and len(cmd) >= 2:
                idx = int(cmd[1])
                serial = list(manager.cameras.keys())[idx]
                manager.start_camera(serial)

            elif action == "rec_start":
                manager.start_record_all()

            elif action == "rec_stop":
                manager.stop_record_all()

            elif action == "help":
                print("""
可用命令:
  status           - 显示所有相机状态
  restart <序号>   - 重启指定相机（序号从 0 开始）
  stop <序号>      - 停止指定相机
  start <序号>     - 启动已停止的相机
  rec_start        - 广播开始所有相机 RAW 录制
  rec_stop         - 广播停止所有相机 RAW 录制
  quit / exit      - 停止所有相机并退出
  help             - 显示此帮助
                """)

            else:
                print(f"未知命令: {action}，输入 'help' 查看可用命令")

        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止所有相机...")
            manager.stop_all()
            break
        except EOFError:
            manager.stop_all()
            break
        except Exception as e:
            print(f"[error] 命令执行失败: {e}")


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

def load_config_from_json(config_path: str) -> tuple[Dict[str, Any], List[CameraConfig]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    global_config: Dict[str, Any] = {
        "server_ip": cfg.get("server_ip", "192.168.1.100"),
        "tsf1_server_port": int(cfg.get("tsf1_server_port", 5001)),
        "hit_server_port": int(cfg.get("hit_server_port", 5003)),
        "shared_options": dict(cfg.get("shared_options", {})),
    }

    camera_configs: List[CameraConfig] = []
    for cam in cfg.get("cameras", []):
        serial = str(cam.get("serial", "")).strip()
        if not serial:
            continue

        alias = str(cam.get("alias", "")).strip()
        enable_tsf1_pub = bool(cam.get("enable_tsf1_pub", True))
        enable_hit_events = bool(cam.get("enable_hit_events", True))

        extra_args = {
            k: v
            for k, v in cam.items()
            if k not in {"serial", "alias", "enabled", "enable_tsf1_pub", "enable_hit_events"}
        }

        camera_configs.append(
            CameraConfig(
                serial=serial,
                alias=alias,
                enable_tsf1_pub=enable_tsf1_pub,
                enable_hit_events=enable_hit_events,
                extra_args=extra_args,
            )
        )

    return global_config, _assign_camera_aliases(camera_configs)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Metavision 多相机启动器（修正版）")
    parser.add_argument("--serials", nargs="+", help="直接指定相机序列号列表")
    parser.add_argument("--config", type=str, help="JSON 配置文件路径")
    parser.add_argument("--auto-detect", action="store_true", help="自动检测所有连接的相机")
    parser.add_argument("--server-ip", type=str, default="192.168.1.100", help="服务器 IP")
    parser.add_argument("--tsf1-server-port", type=int, default=5001, help="TSF1 帧接收端口")
    parser.add_argument("--hit-server-port", type=int, default=5003, help="命中判定接收端口")

    args = parser.parse_args()

    global_config: Dict[str, Any] = {
        "server_ip": args.server_ip,
        "tsf1_server_port": args.tsf1_server_port,
        "hit_server_port": args.hit_server_port,
        "shared_options": {},
    }
    camera_configs: List[CameraConfig] = []

    if args.config:
        global_config, camera_configs = load_config_from_json(args.config)
    elif args.auto_detect:
        serials = detect_metavision_cameras()
        if not serials:
            print("[error] 未检测到任何相机，退出。")
            sys.exit(1)
        camera_configs = _assign_camera_aliases([CameraConfig(serial=s) for s in serials])
    elif args.serials:
        camera_configs = _assign_camera_aliases([CameraConfig(serial=str(s)) for s in args.serials])
    else:
        print("[error] 请提供 --serials、--config 或 --auto-detect 之一")
        parser.print_help()
        sys.exit(1)

    if not camera_configs:
        print("[error] 没有有效的相机配置")
        sys.exit(1)

    print(f"\n准备启动 {len(camera_configs)} 台相机:")
    for i, cfg in enumerate(camera_configs):
        alias = cfg.alias or cfg.serial
        print(f"  [{i}] {_display_name(alias)} (SN={cfg.serial})")

    manager = MultiCameraManager(global_config, camera_configs)

    def signal_handler(sig, frame):
        print("\n收到信号，正在停止所有相机...")
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    manager.start_all()
    interactive_console(manager)


if __name__ == "__main__":
    main()
