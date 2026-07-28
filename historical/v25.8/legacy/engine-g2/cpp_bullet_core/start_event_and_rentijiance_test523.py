#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动：事件相机 A/D + rentijiance 人体检测联动版。

运行位置建议：/home/ysxq/PycharmProjects/Ids_Test_3.9/test607
流程：
  1) 清理 sync_ipc 中旧 ready/trigger/human 文件；
  2) 启动 multi_camera_launcher.py 打开事件相机 A/D，并启用 External Trigger In；
  3) 等待 event_ready_A.flag 和 event_ready_D.flag；
  4) 启动 rentijiance 人体检测，它会打开 MVS_A/MVS_D，配置 Line1=ExposureStartActive，写 human_result_A/D.jsonl；
  5) 任一子进程退出时，自动停止另一个。
"""
import argparse
import datetime
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_ROOT = Path('/home/ysxq/PycharmProjects/Ids_Test_3.9/test607')


def now_stamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def remove_if_exists(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            print(f'[WARN] 删除旧文件失败: {p}: {e}', flush=True)


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'on', 'enable', 'enabled'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off', 'disable', 'disabled'}:
        return False
    return bool(default)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _resolve_arg_path(root: Path, value: str) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (root / p)


def _resolve_runtime_path(root: Path, value: str) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (root / p)


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _build_alias_output_path(base_path: str, alias: str, serial: str, default_ext: str = '') -> str:
    base_path = str(base_path or '').strip()
    if not base_path:
        return ''
    alias = str(alias or '').strip()
    serial = str(serial or '').strip()
    root, ext = os.path.splitext(base_path)
    if not ext and default_ext:
        ext = default_ext
        root = base_path
    if serial and serial in root and alias:
        root = root.replace(serial, alias)
    elif alias and not root.endswith(f'_{alias}') and os.path.basename(root) != alias:
        root = f'{root}_{alias}'
    return root + ext


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_human_config(root: Path, human_config: str) -> Dict[str, Any]:
    cfg = _load_json(_resolve_arg_path(root, human_config))
    active = str(cfg.get('active_profile', '') or '').strip()
    if not active:
        return cfg
    profiles = cfg.get('profiles', {}) or {}
    if not isinstance(profiles, dict) or active not in profiles:
        return cfg
    merged = json.loads(json.dumps(cfg))
    profile_cfg = profiles.get(active) or {}
    if isinstance(profile_cfg, dict):
        _deep_update(merged, profile_cfg)
        merged['active_profile'] = active
    return merged


def _collect_event_files(root: Path, event_config: str) -> tuple[List[Path], List[Path]]:
    cfg = _load_json(_resolve_arg_path(root, event_config))
    ready_files: List[Path] = []
    trigger_files: List[Path] = []
    for cam in cfg.get('cameras', []) or []:
        if not isinstance(cam, dict) or not _as_bool(cam.get('enabled', True), True):
            continue
        if not _as_bool(cam.get('enable_ext_trigger_in', False), False):
            continue
        alias = str(cam.get('alias', '') or '').strip()
        serial = str(cam.get('serial', '') or '').strip()
        if not alias:
            alias = serial
        ready_arg = str(cam.get('ext_trigger_ready_file', '') or '').strip()
        if ready_arg:
            ready = _build_alias_output_path(ready_arg, alias, serial, default_ext='.flag')
            ready_files.append(_resolve_runtime_path(root, ready))
        log_arg = str(cam.get('ext_trigger_log', '') or '').strip()
        if log_arg:
            trig = _build_alias_output_path(log_arg, alias, serial, default_ext='.csv')
            trigger_files.append(_resolve_runtime_path(root, trig))
    return _unique_paths(ready_files), _unique_paths(trigger_files)


def _sync_template_path(root: Path, sync_cfg: Dict[str, Any], key: str, camera: str, default_template: str) -> Path:
    sync_root = str(sync_cfg.get('root', str(root / 'sync_ipc')) or str(root / 'sync_ipc'))
    template = str(sync_cfg.get(key, default_template) or default_template)
    path = template.format(camera=str(camera), cam=str(camera), root=sync_root)
    return _resolve_runtime_path(root, path)


def _collect_human_files(root: Path, human_config: str) -> tuple[List[Path], List[Path]]:
    cfg = _load_human_config(root, human_config)
    sync_cfg = cfg.get('sync', {}) or {}
    routes = cfg.get('routes', []) or []
    ready_files: List[Path] = []
    human_files: List[Path] = []
    for route in routes:
        if not isinstance(route, dict) or not _as_bool(route.get('enabled', True), True):
            continue
        if str(route.get('source', '') or 'hik') != 'hik':
            continue
        cid = str(route.get('id', route.get('name', '')) or '').strip()
        if not cid:
            continue
        ready_files.append(_sync_template_path(root, sync_cfg, 'ready_file_template', cid, '{root}/event_ready_{camera}.flag'))
        human_files.append(_sync_template_path(root, sync_cfg, 'human_jsonl_template', cid, '{root}/human_result_{camera}.jsonl'))
    return _unique_paths(ready_files), _unique_paths(human_files)


def start_proc(label: str, cmd: List[str], log_path: Path, cwd: Path, pipe_stdin: bool = False) -> subprocess.Popen:
    mkdirp(log_path.parent)
    log_fp = open(log_path, 'w', encoding='utf-8', buffering=1)
    print(f'[{label}] start: ' + ' '.join(cmd), flush=True)
    print(f'[{label}] log: {log_path}', flush=True)
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    # 保留引用，避免文件句柄被回收
    p._log_fp = log_fp  # type: ignore[attr-defined]
    return p


def wait_ready(files: List[Path], timeout_s: float, poll_s: float = 0.1) -> None:
    deadline: Optional[float] = None if timeout_s <= 0 else time.time() + timeout_s
    print('[launcher] waiting event ready flags:', flush=True)
    for f in files:
        print(f'  - {f}', flush=True)
    while True:
        missing = [f for f in files if not f.exists()]
        if not missing:
            print('[launcher] all event ready flags detected.', flush=True)
            return
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError('event ready timeout, missing: ' + ', '.join(str(f) for f in missing))
        time.sleep(poll_s)


def stop_proc(label: str, p: Optional[subprocess.Popen], grace_s: float = 4.0) -> None:
    if p is None or p.poll() is not None:
        return
    print(f'[launcher] stopping {label} pid={p.pid}', flush=True)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        try:
            p.send_signal(signal.SIGINT)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(0.5)
    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def tail_file(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(lines[-n:])
    except Exception as e:
        return f'<无法读取 {path}: {e}>'


def write_record_command(command_file: Path, cmd: str, session_id: Optional[str] = None, source: str = 'launcher_terminal') -> bool:
    payload = {
        'seq': int(time.time() * 1000),
        'cmd': str(cmd).lower(),
        'wall_us': int(time.time() * 1_000_000),
        'source': source,
    }
    if session_id:
        payload['session_id'] = str(session_id)
    try:
        mkdirp(command_file.parent)
        tmp = command_file.with_suffix(command_file.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        tmp.replace(command_file)
        print(f'[record] {cmd} command -> {command_file} session={payload.get("session_id", "")}', flush=True)
        return True
    except Exception as e:
        print(f'[record][ERROR] 写录制控制文件失败: {command_file}: {e}', flush=True)
        return False


def _poll_terminal_command() -> str:
    # 非阻塞读取终端命令。为了不破坏当前 OpenCV 窗口热键，这里要求在终端输入 s 后回车。
    # 事件相机/MVS 窗口里按 S 也会写同一个控制文件。
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return ''
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return ''
        return sys.stdin.readline().strip().lower()
    except Exception:
        return ''


def main() -> int:
    ap = argparse.ArgumentParser('一键启动事件相机 + rentijiance 人体检测联动版')
    ap.add_argument('--root', default=str(DEFAULT_ROOT), help='工程根目录，默认 test607')
    ap.add_argument('--python', default=sys.executable, help='Python 解释器，默认当前环境')
    ap.add_argument('--event-config', default='cameras_sync_live_D_v3_ts_align.json')
    ap.add_argument('--human-config', default='rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json')
    ap.add_argument('--event-launcher', default='multi_camera_launcher.py')
    ap.add_argument('--human-script', default='rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py')
    ap.add_argument('--ready-timeout-s', type=float, default=120.0)
    ap.add_argument('--clean-sync', action='store_true', default=True)
    ap.add_argument('--no-clean-sync', dest='clean_sync', action='store_false')
    ap.add_argument('--event-only', action='store_true')
    ap.add_argument('--human-only', action='store_true')
    ap.add_argument('--enable-one-key-record', action='store_true', default=True)
    ap.add_argument('--disable-one-key-record', dest='enable_one_key_record', action='store_false')
    ap.add_argument('--record-command-file', default='', help='统一录制控制文件；默认 root/sync_ipc/record_control.json')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    sync_ipc = root / 'sync_ipc'
    record_command_file = Path(args.record_command_file).expanduser().resolve() if args.record_command_file else (sync_ipc / 'record_control.json')
    log_dir = root / 'launcher_logs' / f'event_rentijiance_{now_stamp()}'
    mkdirp(sync_ipc)
    mkdirp(log_dir)

    event_ready_files, trigger_files = _collect_event_files(root, args.event_config)
    human_ready_files, human_files = _collect_human_files(root, args.human_config)
    ready_files = human_ready_files or event_ready_files
    if not ready_files:
        ready_files = [sync_ipc / 'event_ready_A.flag', sync_ipc / 'event_ready_D.flag']
    if event_ready_files and human_ready_files and {str(p) for p in event_ready_files} != {str(p) for p in human_ready_files}:
        print('[launcher][WARN] event config ready files and human sync ready files differ.', flush=True)
        print('[launcher][WARN] human process waits for:', flush=True)
        for f in human_ready_files:
            print(f'  - {f}', flush=True)
        print('[launcher][WARN] event process writes:', flush=True)
        for f in event_ready_files:
            print(f'  - {f}', flush=True)

    if args.clean_sync:
        remove_if_exists(_unique_paths(ready_files + event_ready_files + human_ready_files + trigger_files + human_files + [record_command_file]))

    event_p = None
    human_p = None
    try:
        if not args.human_only:
            event_cmd = [args.python, str(root / args.event_launcher), '--config', str(root / args.event_config)]
            event_p = start_proc('event', event_cmd, log_dir / 'event_camera.log', root, pipe_stdin=True)
            if not args.event_only:
                wait_ready(ready_files, timeout_s=float(args.ready_timeout_s))
        if not args.event_only:
            if args.human_only:
                wait_ready(ready_files, timeout_s=float(args.ready_timeout_s))
            human_cmd = [args.python, str(root / args.human_script), '--config', str(root / args.human_config)]
            human_p = start_proc('human', human_cmd, log_dir / 'rentijiance.log', root, pipe_stdin=False)

        print('\n[launcher] started. Ctrl+C 停止全部进程。', flush=True)
        print(f'[launcher] logs: {log_dir}', flush=True)
        print(f'[launcher] sync_ipc: {sync_ipc}', flush=True)
        if args.enable_one_key_record:
            print(f'[launcher] one-key record control: {record_command_file}', flush=True)
            print('[launcher] 录制热键：在事件/MVS窗口按 S，或在本终端输入 s 后回车，开始/停止所有启用的一键录制。', flush=True)
        launcher_recording = False
        while True:
            time.sleep(1.0)
            term_cmd = _poll_terminal_command() if args.enable_one_key_record else ''
            if term_cmd in ('s', 'record', 'toggle'):
                if launcher_recording:
                    if write_record_command(record_command_file, 'stop'):
                        launcher_recording = False
                else:
                    sid = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    if write_record_command(record_command_file, 'start', session_id=sid):
                        launcher_recording = True
            elif term_cmd in ('rec_start', 'start_record'):
                sid = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                if write_record_command(record_command_file, 'start', session_id=sid):
                    launcher_recording = True
            elif term_cmd in ('rec_stop', 'stop_record'):
                if write_record_command(record_command_file, 'stop'):
                    launcher_recording = False
            if event_p is not None and event_p.poll() is not None:
                print(f'[launcher][ERROR] event process exited: code={event_p.returncode}', flush=True)
                print('----- event log tail -----')
                print(tail_file(log_dir / 'event_camera.log'))
                return int(event_p.returncode or 1)
            if human_p is not None and human_p.poll() is not None:
                print(f'[launcher][ERROR] human process exited: code={human_p.returncode}', flush=True)
                print('----- human log tail -----')
                print(tail_file(log_dir / 'rentijiance.log'))
                return int(human_p.returncode or 1)
    except KeyboardInterrupt:
        print('\n[launcher] Ctrl+C received.', flush=True)
        return 130
    except Exception as e:
        print(f'[launcher][ERROR] {type(e).__name__}: {e}', flush=True)
        print('----- event log tail -----')
        print(tail_file(log_dir / 'event_camera.log'))
        print('----- human log tail -----')
        print(tail_file(log_dir / 'rentijiance.log'))
        return 1
    finally:
        try:
            if 'record_command_file' in locals() and args.enable_one_key_record:
                write_record_command(record_command_file, 'stop', source='launcher_exit')
        except Exception:
            pass
        stop_proc('human', human_p)
        stop_proc('event', event_p)


if __name__ == '__main__':
    raise SystemExit(main())
