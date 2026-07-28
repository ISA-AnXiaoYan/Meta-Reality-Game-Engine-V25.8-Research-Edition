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
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


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

    one_key_record = mvs_cfg.setdefault('one_key_record', {})
    one_key_record['enable'] = True
    one_key_record['command_file'] = record_cmd_path
    one_key_record['status_file'] = record_status_path
    # ---- YSXQ one-key record path fix end ----

    # Ensure the event-side non-blocking human mask prefilter reads the same MVS
    # config that the MVS/YOLO process will actually use, and use the same
    # absolute human JSONL template to avoid cwd-dependent lookup.
    shared = event_cfg.setdefault('shared_options', {})
    shared['event_human_mask_mvs_config'] = str(mvs_path)
    shared['event_human_mask_jsonl_template'] = str(root_path.resolve() / 'human_result_{camera}.jsonl')
    _inject_event_sync_start_options(event_cfg, project_root)

    event_path.write_text(json.dumps(event_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    mvs_path.write_text(json.dumps(mvs_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[launcher] unified config={unified_path}", flush=True)
    print(f"[launcher] runtime event config={event_path}", flush=True)
    print(f"[launcher] runtime mvs config={mvs_path}", flush=True)
    return event_path, mvs_path, event_cfg

def parse_cams(args_cams: str, event_cfg: Dict) -> List[str]:
    if args_cams:
        return [x.strip() for x in args_cams.split(',') if x.strip()]
    cams = []
    for c in event_cfg.get('cameras', []):
        if c.get('alias') and c.get('enabled', True):
            cams.append(str(c['alias']).strip())
    return cams or list('ABCDE')


def wait_ready(root: Path, cams: List[str], timeout: float, not_before_wall: float = 0.0) -> bool:
    """Wait until every event camera writes a fresh ready flag.

    not_before_wall prevents stale event_ready_*.flag files from a previous run
    from satisfying readiness before the new event process has actually started.
    """
    deadline = time.time() + float(timeout)
    missing = []
    while time.time() < deadline:
        missing = []
        for c in cams:
            flag = root / f'event_ready_{c}.flag'
            if not flag.exists():
                missing.append(c)
                continue
            if float(not_before_wall or 0.0) > 0.0:
                try:
                    if flag.stat().st_mtime < float(not_before_wall) - 0.001:
                        missing.append(c)
                        continue
                except Exception:
                    missing.append(c)
                    continue
        if not missing:
            print(f"[launcher] fresh event ready flags OK: {','.join(cams)}", flush=True)
            return True
        time.sleep(0.2)
    print(f"[launcher][WARN] event ready timeout; missing/stale: {','.join(missing)}", flush=True)
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
    for name in ('sync_start.flag',):
        path = root / name
        if path.exists():
            try:
                path.unlink()
                print(f"[launcher] removed stale event sync file {path}", flush=True)
            except Exception as exc:
                print(f"[launcher][WARN] cannot remove stale event sync file {path}: {exc}", flush=True)
    for c in cams:
        for name in (f'event_ready_{c}.flag', f'event_trigger_{c}.csv', f'mvs_frame_audit_{c}.csv'):
            path = root / name
            if path.exists():
                try:
                    path.unlink()
                    print(f"[launcher] removed stale event sync file {path}", flush=True)
                except Exception as exc:
                    print(f"[launcher][WARN] cannot remove stale event sync file {path}: {exc}", flush=True)


def cleanup_stale(project_root: Path, cams: List[str], clean_logs: bool = False) -> None:
    for prefix in ['mvs_latest_', 'event_overlay_latest_']:
        for c in cams:
            for p in Path('/dev/shm').glob(f'{prefix}{c}*'):
                try:
                    p.unlink()
                    print(f"[launcher] removed stale {p}", flush=True)
                except Exception as exc:
                    print(f"[launcher][WARN] cannot remove {p}: {exc}", flush=True)
    for fn in ['record_control.json', 'record_status.json']:
        p = project_root / 'sync_ipc' / fn
        if p.exists():
            try:
                p.unlink()
                print(f"[launcher] removed {p}", flush=True)
            except Exception:
                pass
    if clean_logs:
        for pat in ['human_result_*.jsonl','hit_candidate_*.jsonl','hit_judge_debug_*.jsonl','overlay_bullet_point_*.jsonl','overlay_bullet_point_all.jsonl','overlay_bullet_event_*.jsonl','overlay_bullet_event_all.jsonl','mvs_frame_audit_*.csv','frame_bundle_audit_*.csv']:
            for p in (project_root/'sync_ipc').glob(pat):
                try:
                    p.unlink()
                except Exception:
                    pass


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

def main() -> int:
    ap = argparse.ArgumentParser(description='One-click launcher for event/MVS/hit/OpenGL fusion stack')
    ap.add_argument('--project-root', default='', help='Project root. Default: directory containing this file')
    ap.add_argument('--cams', default='', help='Comma separated camera ids. Default: enabled event cameras, usually A,B,C,D,E')
    ap.add_argument('--config', default='ids_8cam_fusion_config.json', help='Single unified fusion JSON. Default: ids_8cam_fusion_config.json. Set empty string to use --event-config/--mvs-config directly.')
    ap.add_argument('--event-config', default='cameras_sync_live_D_v3_ts_align.json')
    ap.add_argument('--mvs-config', default='hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json')
    ap.add_argument('--preview-fps', type=float, default=60.0)
    ap.add_argument('--hit-port', type=int, default=5003)
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
    ap.add_argument('--mvs-aggregator-delay', type=float, default=0.8, help='Delay after starting coordinator before starting MVS/YOLO aggregator.')
    ap.add_argument('--mvs-worker-cpus', default='', help='Optional comma-separated CPU cores for workers, e.g. 3,4,5,6,7,8,9,10.')
    ap.add_argument('--mvs-coordinator-cpu', default='', help='Optional CPU core for trigger coordinator, e.g. 2.')
    ap.add_argument('--clean-shm', action='store_true', help='Remove stale /dev/shm frames before start')
    ap.add_argument('--clean-jsonl', action='store_true', help='Remove old human/hit debug jsonl files before start')
    ap.add_argument('--ready-timeout', type=float, default=30.0)
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
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parent
    os.chdir(project_root)
    print(f"[launcher] project_root={project_root}", flush=True)

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
    cams = parse_cams(args.cams, event_cfg)
    cam_arg = ','.join(cams)
    print(f"[launcher] cams={cam_arg}", flush=True)

    (project_root / 'sync_ipc').mkdir(exist_ok=True)
    if args.clean_shm or args.clean_jsonl:
        cleanup_stale(project_root, cams, clean_logs=args.clean_jsonl)

    py = sys.executable or 'python3'

    child_log_dir = Path(args.child_log_dir).expanduser()
    if not child_log_dir.is_absolute():
        child_log_dir = project_root / child_log_dir
    child_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] child logs dir={child_log_dir}", flush=True)
    print(
        f"[launcher] console_log_mode={args.console_log_mode} startup_log_lines={args.startup_log_lines}",
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
    trigger_coord_script = resolve_script(project_root, ['mvs_trigger_coordinator.py'])
    mvs_worker_script = resolve_script(project_root, ['mvs_cam_worker_external_trigger.py'])

    procs: List[ManagedProc] = []
    stopping = False

    def make_proc(name: str, cmd: List[str], critical: bool = True) -> ManagedProc:
        return ManagedProc(
            name,
            cmd,
            project_root,
            critical=critical,
            log_dir=child_log_dir,
            console_mode=args.console_log_mode,
            startup_lines=args.startup_log_lines,
        )

    def stop_all(signum=None, frame=None):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("\n[launcher] stopping all processes...", flush=True)
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
            wait_ready(project_root / 'sync_ipc', cams, args.ready_timeout, not_before_wall=event_start_wall)
            if args.event_ready_settle_delay > 0:
                time.sleep(args.event_ready_settle_delay)
        if args.mvs_delay > 0:
            time.sleep(args.mvs_delay)
        if not args.skip_mvs:
            if args.mvs_split_workers:
                _enable_external_mvs_shm_mode(Path(args.mvs_config), project_root)
                worker_cpus = str(args.mvs_worker_cpus or '')
                for idx, cid in enumerate(cams):
                    cpu = _split_worker_cpu(worker_cpus, idx)
                    worker_cmd = [py, mvs_worker_script, '--config', args.mvs_config, '--cam', str(cid)]
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
                ]
                if float(args.mvs_trigger_fps or 0.0) > 0.0:
                    coord_cmd += ['--fps', str(args.mvs_trigger_fps)]
                coord_cmd = _taskset_cmd(str(args.mvs_coordinator_cpu or ''), coord_cmd)
                p = make_proc('TRIGGER', coord_cmd, critical=True)
                procs.append(p); p.start()
                if args.mvs_aggregator_delay > 0:
                    time.sleep(args.mvs_aggregator_delay)
                p = make_proc('MVS_AGG', [py, mvs_script, '--config', args.mvs_config], critical=True)
                procs.append(p); p.start()
            else:
                p = make_proc('MVS', [py, mvs_script, '--config', args.mvs_config], critical=True)
                procs.append(p); p.start()
        if args.hit_delay > 0:
            time.sleep(args.hit_delay)
        if not args.skip_hit:
            hit_cmd = [py, hit_script, '--config', args.mvs_config, '--port', str(args.hit_port), '--cams', cam_arg]
            if args.overlay_ws_enable:
                hit_cmd += ['--overlay-ws-enable', '--overlay-ws-host', str(args.overlay_ws_host), '--overlay-ws-port', str(args.overlay_ws_port)]
            p = make_proc('HIT', hit_cmd, critical=True)
            procs.append(p); p.start()
        if args.gl_delay > 0:
            time.sleep(args.gl_delay)
        if not args.skip_gl:
            # GL is preview-only. Closing the window should not necessarily kill
            # camera capture / hit judge, but EVENT/MVS/HIT remain critical.
            p = make_proc('GL', [py, gl_script, '--config', args.mvs_config, '--cams', cam_arg, '--preview-fps', str(args.preview_fps)], critical=False)
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
                    print(f"[launcher][{level}] {p.name} exited rc={rc} critical={p.critical}", flush=True)
                if p.critical:
                    print(f"[launcher][ERROR] critical process {p.name} exited; stopping whole stack", flush=True)
                    stop_all()
                    return 1
            if not any_running:
                print('[launcher] all child processes exited.', flush=True)
                break
            time.sleep(0.5)
    finally:
        stop_all()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
