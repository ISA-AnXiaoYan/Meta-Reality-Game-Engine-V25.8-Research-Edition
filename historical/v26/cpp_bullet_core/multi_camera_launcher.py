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
    """
    arg_name = _normalize_arg_name(key)

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

    # 这些已经显式加过，不再从 shared_options / extra_args 重复加
    skip_keys = {
        "enable_tsf1_pub",
        "tsf1_auto_start_sender",
        "enable_hit_events",
        "tsf1_pub_name",
        "tsf1_camera_id",
        "tsf1_server_ip",
        "tsf1_server_port",
        "hit_server_ip",
        "hit_server_port",
        "hit_camera_alias",
        "enabled",
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
        self.script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py",
        )
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"找不到主脚本: {self.script_path}")

        for cfg in camera_configs:
            self.cameras[cfg.serial] = CameraProcess(config=cfg)

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

        serials = list(self.cameras.keys())
        for i, serial in enumerate(serials):
            ok = self.start_camera(serial)
            if not ok:
                failed.append(serial)
                continue

            cp = self.cameras[serial]
            alias = cp.config.alias or serial
            print(f"  等待相机 {_display_name(alias)} 初始化 (最多 {max_wait_per_cam}s)...")

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
            else:
                print(f"  [OK] 相机 {_display_name(alias)} (SN={serial}) 启动成功")
                success.append(serial)

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

    def stop_all(self) -> None:
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