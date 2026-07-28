import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for _p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent / "test607", _THIS_DIR.parent / "test", _THIS_DIR.parent / "test-42"]:
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)


# Ensure Metavision SDK Python modules are visible when this script is launched
# by launch_fusion_system.py or shell scripts that do not inherit the same
# sys.path as the interactive conda terminal. On this machine Metavision is
# provided from /usr/lib/python3/dist-packages, not from conda site-packages.
_METAVISION_DIST_PATHS = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
)
for _p in _METAVISION_DIST_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

if os.environ.get("YSXQ_EVENT_ENV_DEBUG", "0") == "1":
    print("[EVENT_ENV] python:", sys.executable, flush=True)
    print("[EVENT_ENV] PYTHONPATH:", os.environ.get("PYTHONPATH", ""), flush=True)
    print("[EVENT_ENV] sys.path:", sys.path, flush=True)

import csv
import inspect
import json
import cv2
import numpy as np

from latest_frame_shm import SharedFrameWriter
from obs_srt_streamer import ObsSrtStreamer
import os
import subprocess
import sys
import datetime
import time
import threading
import queue
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_analytics import SpatterTrackerAlgorithm, SpatterTrackingConfig, EventSpatterClusterBuffer
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

from bullet_event_sender import BulletEventSender, BulletEventExtractor
from line_motion_filter_backfill import (
    LinearMotionFilter,
    TrackConfig,
    AssociationConfig,
    DrawConfig,
)


ev_filter_type_dict = {
    "NoFilter":       SpatterTrackingConfig.FilterType.NoFilter,
    "FilterNegative": SpatterTrackingConfig.FilterType.FilterNegative,
    "FilterPositive": SpatterTrackingConfig.FilterType.FilterPositive,
    "SeparateFilter": SpatterTrackingConfig.FilterType.SeparateFilter,
}


class AdaptiveEventThrottler:
    """
    自适应事件限流器 (MVP Version)
    
    策略：
    1. 监控每个 slice 的事件数量。
    2. 超过阈值时启用等间隔采样 (Stride Sampling)。
    3. 使用 Burst Hold 机制防止状态频繁切换。
    """
    def __init__(self, max_events_per_slice=30000, severe_events_per_slice=60000, burst_hold_ms=1500):
        self.max_eps = max_events_per_slice
        self.severe_eps = severe_events_per_slice
        self.burst_hold_us = burst_hold_ms * 1000
        self.is_throttled = False
        self.throttle_start_ts = 0
        self.state = "NORMAL" # NORMAL, LIGHT, SEVERE, HOLD

    def process_batch(self, evs, current_ts):
        n = len(evs)
        if n <= self.max_eps:
            # 恢复正常逻辑
            if self.state != "NORMAL":
                if current_ts - self.throttle_start_ts > self.burst_hold_us:
                    self.state = "NORMAL"
                    self.is_throttled = False
                    print(f"[Throttler] Recovered to NORMAL at ts={current_ts}")
            return evs

        # 触发限流
        if self.state == "NORMAL":
            self.throttle_start_ts = current_ts
            
        self.is_throttled = True
        
        if n > self.severe_eps:
            self.state = "SEVERE"
            step = int(np.ceil(n / (self.max_eps * 0.5))) # 更激进的采样
        else:
            self.state = "LIGHT"
            step = int(np.ceil(n / self.max_eps))
            
        if step > 1:
            return evs[::step]
        return evs


class CameraShakeGuard:
    """
    基于代码实际收到的 raw 事件率判断“相机抖动/全局运动过强”。
    一旦触发，在一小段时间内直接跳过重检测流程，保证画面不卡。
    """
    def __init__(self, enabled=True, rate_thresh_mevs=14.0, trigger_slices=2, hold_ms=800, release_ratio=0.85):
        self.enabled = bool(enabled)
        self.rate_thresh_mevs = float(rate_thresh_mevs)
        self.trigger_slices = max(1, int(trigger_slices))
        self.hold_us = max(0, int(hold_ms) * 1000)
        self.release_ratio = float(max(0.1, min(1.0, release_ratio)))

        self._over_count = 0
        self._active = False
        self._hold_until_ts = -1

    @property
    def active(self):
        return self._active

    def update(self, raw_n, dt_us, ts):
        if not self.enabled:
            return False, float(raw_n) / float(max(1, dt_us)), False, False

        rate_mevs = float(raw_n) / float(max(1, dt_us))
        entered = False
        exited = False

        if rate_mevs >= self.rate_thresh_mevs:
            self._over_count += 1
        else:
            self._over_count = 0

        if (not self._active) and self._over_count >= self.trigger_slices:
            self._active = True
            self._hold_until_ts = int(ts) + self.hold_us
            entered = True

        if self._active:
            if rate_mevs >= self.rate_thresh_mevs:
                self._hold_until_ts = int(ts) + self.hold_us
            elif int(ts) >= self._hold_until_ts and rate_mevs <= self.rate_thresh_mevs * self.release_ratio:
                self._active = False
                self._over_count = 0
                exited = True

        return self._active, rate_mevs, entered, exited


class HotkeyRawRecorder:
    def __init__(self, device, enabled=False, output_dir="./raw_captures", prefix="capture", camera_id="cam0"):
        self.device = device
        self.enabled = bool(enabled)
        self.output_dir = str(output_dir)
        self.prefix = str(prefix)
        self.camera_id = str(camera_id)
        self.recording = False
        self.current_path = None
        self._current_session_id = None
        self._closed = False
        self._lock = threading.Lock()

    def can_record(self):
        return self.enabled and self.device is not None and getattr(self.device, 'get_i_events_stream', None) is not None and self.device.get_i_events_stream() is not None

    def is_recording(self):
        with self._lock:
            return bool(self.recording)

    def _build_path(self, session_id=None):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        stem = session_id if session_id else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return str(Path(self.output_dir) / f"{self.prefix}_{stem}_{self.camera_id}.raw")

    def start(self, session_id=None):
        with self._lock:
            if self._closed:
                print('[RAW] start ignored: recorder is closing/closed.', flush=True)
                return False
            if not self.can_record():
                print('[RAW] recorder unavailable on this device.', flush=True)
                return False
            if self.recording:
                if session_id and session_id != self._current_session_id:
                    print(
                        f'[RAW] start ignored: already recording -> {self.current_path} '
                        f'(current_session={self._current_session_id}, new_session={session_id})',
                        flush=True,
                    )
                return True
            path = self._build_path(session_id=session_id)
            self.device.get_i_events_stream().log_raw_data(path)
            self.recording = True
            self.current_path = path
            self._current_session_id = session_id
            print(f'[RAW] recording started -> {path}', flush=True)
            return True

    def stop(self):
        with self._lock:
            if not self.recording:
                return False
            try:
                self.device.get_i_events_stream().stop_log_raw_data()
            finally:
                path = self.current_path
                self.recording = False
                self.current_path = None
                self._current_session_id = None
                print(f'[RAW] recording stopped -> {path}', flush=True)
            return True

    def toggle(self, session_id=None):
        with self._lock:
            is_recording = bool(self.recording)
        if is_recording:
            return self.stop()
        return self.start(session_id=session_id)

    def close(self):
        with self._lock:
            self._closed = True
            is_recording = bool(self.recording)
        if is_recording:
            self.stop()


class RemoteRawRecordController:
    def __init__(self, enabled=False, command_file='/tmp/eventcam_raw_record_cmd.json', poll_interval_ms=50):
        self.enabled = bool(enabled)
        self.command_file = str(command_file)
        self.poll_interval_s = max(0.01, float(poll_interval_ms) / 1000.0)
        self._last_poll_wall = 0.0
        self._last_seq = None

    def write_command(self, cmd, session_id=None):
        if not self.enabled:
            return False
        payload = {
            'seq': int(time.time() * 1000),
            'cmd': str(cmd).lower(),
        }
        if session_id:
            payload['session_id'] = str(session_id)
        path = Path(self.command_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            return True
        except OSError as exc:
            print(f'[RAW] write command failed: {path} -> {exc}. 请检查磁盘空间: df -h /tmp /home', flush=True)
            return False

    def poll(self):
        if not self.enabled:
            return None
        now = time.time()
        if now - self._last_poll_wall < self.poll_interval_s:
            return None
        self._last_poll_wall = now
        path = Path(self.command_file)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None
        seq = payload.get('seq')
        if seq is None or seq == self._last_seq:
            return None
        self._last_seq = seq
        return payload


def _record_payload_targets_include(payload, target: str) -> bool:
    """Return True when record_control.json targets this recorder.

    Backward compatible: missing targets means broadcast to all recorders.
    Supported aliases:
      event_raw: event, events, raw, eventcam, event_camera
      mvs_raw: mvs, mvs_video, hik
    """
    if not isinstance(payload, dict):
        return True
    targets = payload.get('targets', payload.get('target', payload.get('record_targets', None)))
    if targets is None or targets == "":
        return True
    if isinstance(targets, str):
        items = [x.strip() for x in targets.replace(';', ',').split(',')]
    elif isinstance(targets, (list, tuple, set)):
        items = [str(x).strip() for x in targets]
    else:
        items = [str(targets).strip()]

    norm = {x.lower().replace('-', '_') for x in items if x}
    if not norm or norm & {'all', 'both', '*'}:
        return True

    target = str(target).lower().replace('-', '_')
    aliases = {
        'event_raw': {'event_raw', 'event', 'events', 'raw', 'raw_event', 'eventcam', 'event_camera'},
        'mvs_raw': {'mvs_raw', 'mvs', 'mvs_video', 'video', 'hik', 'hik_mvs'},
    }
    return bool(norm & aliases.get(target, {target}))


class AsyncRawRecordControlService:
    """
    将 RAW 录制控制从重检测主循环中解耦出来。

    - 远程广播命令文件轮询在独立线程中完成
    - 本地热键请求也统一走同一控制线程
    - 真正调用 log_raw_data / stop_log_raw_data 的入口只有这里一处
    """
    def __init__(self, recorder: HotkeyRawRecorder, remote_controller: RemoteRawRecordController | None = None):
        self.recorder = recorder
        self.remote_controller = remote_controller
        self._local_requests: queue.Queue = queue.Queue(maxsize=64)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f'RawRecordControl-{self.recorder.camera_id}',
        )
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _submit_local_request(self, cmd, session_id=None, source='local'):
        if self._stop_evt.is_set():
            return False
        try:
            self._local_requests.put_nowait((cmd, session_id, source))
            return True
        except queue.Full:
            print(f'[RAW] async control queue full, drop cmd={cmd} source={source}', flush=True)
            return False

    def request_start(self, session_id=None, source='local'):
        return self._submit_local_request('start', session_id, source)

    def request_stop(self, source='local'):
        return self._submit_local_request('stop', None, source)

    def request_toggle(self, session_id=None, source='local'):
        return self._submit_local_request('toggle', session_id, source)

    def _handle_command(self, cmd, session_id=None, source='local'):
        if self._stop_evt.is_set():
            return
        cmd = str(cmd).lower()
        if cmd == 'start':
            self.recorder.start(session_id=session_id)
        elif cmd == 'stop':
            self.recorder.stop()
        elif cmd == 'toggle':
            self.recorder.toggle(session_id=session_id)
        elif cmd:
            print(f'[RAW] unknown async control cmd={cmd} source={source}', flush=True)

    def _poll_remote_once(self):
        if self._stop_evt.is_set() or self.remote_controller is None:
            return
        payload = self.remote_controller.poll()
        if self._stop_evt.is_set() or payload is None:
            return
        if not _record_payload_targets_include(payload, 'event_raw'):
            return
        cmd = str(payload.get('cmd', '')).lower()
        session_id = payload.get('session_id')
        self._handle_command(cmd, session_id=session_id, source=str(payload.get('source', 'remote') or 'remote'))

    def _drain_local_requests(self):
        processed = False
        while not self._stop_evt.is_set():
            try:
                cmd, session_id, source = self._local_requests.get_nowait()
            except queue.Empty:
                break
            self._handle_command(cmd, session_id=session_id, source=source)
            processed = True
        return processed

    def _worker(self):
        poll_timeout = 0.05
        if self.remote_controller is not None:
            poll_timeout = max(0.005, min(0.05, float(self.remote_controller.poll_interval_s)))
        while not self._stop_evt.is_set():
            try:
                self._poll_remote_once()
                try:
                    cmd, session_id, source = self._local_requests.get(timeout=poll_timeout)
                except queue.Empty:
                    continue
                if self._stop_evt.is_set():
                    break
                self._handle_command(cmd, session_id=session_id, source=source)
                self._drain_local_requests()
            except Exception as exc:
                print(f'[RAW] async control service error: {exc}', flush=True)
                time.sleep(0.05)

    def close(self):
        self._stop_evt.set()
        if self._started:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print('[RAW] async control service did not stop within 2s; recorder will reject new starts.', flush=True)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Spatter Tracking + line filter + live-camera HAL tuning (crop/AFK/trail/ERC/biases) + effective-args diagnostics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    base_options.add_argument('-s', '--process-from', dest='process_from', type=int, default=0)
    base_options.add_argument('-e', '--process-to', dest='process_to', type=int, default=None)
    base_options.add_argument('--stc-threshold', dest='stc_threshold', type=int, default=0)
    base_options.add_argument('--headless', dest='headless', action='store_true', default=False)
    base_options.add_argument('--render-every-n', dest='render_every_n', type=int, default=10)

    algo_options = parser.add_argument_group('Spatter options')
    algo_options.add_argument('--polarity-filter', dest='filtered_polarity_str', type=str,
                              default="NoFilter", choices=["NoFilter", "FilterNegative", "FilterPositive", "SeparateFilter"])
    algo_options.add_argument('--cell-width',               dest='cell_width',           type=int,   default=5)
    algo_options.add_argument('--cell-height',              dest='cell_height',          type=int,   default=5)
    algo_options.add_argument('--processing-accumulation-time', dest='accumulation_time_us', type=int, default=3000)
    algo_options.add_argument('--static-memory',            dest='static_memory_us',     type=int,   default=1000000)
    algo_options.add_argument('--untracked-threshold',      dest='untracked_ths',        type=int,   default=5)
    algo_options.add_argument('--min-track-time',           dest='min_track_time',       type=int,   default=1000)
    algo_options.add_argument('-a', '--activation-threshold', dest='activation_ths',     type=int,   default=10)
    algo_options.add_argument('--apply-filter',             dest='apply_filter',         default=False, action='store_true')
    algo_options.add_argument('-D', '--max-distance',       dest='max_distance',         type=int,   default=80)
    algo_options.add_argument('--max-size-variation',       dest='max_size_variation',   type=int,   default=100)
    algo_options.add_argument('--min-dist-moving-obj',      dest='min_dist_moving_obj',  type=int,   default=3)
    algo_options.add_argument('--min-size',                 dest='min_size',             type=int,   default=5)
    algo_options.add_argument('--max-size',                 dest='max_size',             type=int,   default=100)
    algo_options.add_argument('--nozone', nargs='+', action='append', type=int,
                              metavar='center_x center_y radius inside', default=[])

    line_options = parser.add_argument_group('Realtime line filter options')
    line_options.add_argument('--use-cpp-line-filter', dest='use_cpp_line_filter', action='store_true', default=False,
                              help='使用全手写 C++ CppLinearMotionFilter 最终实现替代 Python LinearMotionFilter；检测流程/输出接口保持一致。')
    line_options.add_argument('--disable-cpp-line-filter', dest='use_cpp_line_filter', action='store_false',
                              help='强制使用 Python 原版 LinearMotionFilter。')
    line_options.add_argument('--line-history-len',              dest='line_history_len',              type=int,   default=8)
    line_options.add_argument('--line-draw-history-len',         dest='line_draw_history_len',         type=int,   default=40)
    line_options.add_argument('--line-pre-confirm-history-len',  dest='line_pre_confirm_history_len',  type=int,   default=24)
    line_options.add_argument('--line-draw-hold-missed-frames',  dest='line_draw_hold_missed_frames',  type=int,   default=2)
    line_options.add_argument('--line-draw-connect-max-gap-px',  dest='line_draw_connect_max_gap_px',  type=float, default=72.0)
    line_options.add_argument('--disable-line-draw-kinematic-fit', dest='line_draw_kinematic_fit_enabled',
                              action='store_false', default=True,
                              help='禁用 P7 运动学拟合绘制，退回直接连接检测框中心点')
    line_options.add_argument('--line-draw-kinematic-residual-px', dest='line_draw_kinematic_residual_px', type=float, default=0.0,
                              help='P7 运动学拟合残差阈值 px；0=自动，越小越不相信偏离点')
    line_options.add_argument('--line-draw-kinematic-max-curve-px', dest='line_draw_kinematic_max_curve_px', type=float, default=32.0,
                              help='P7 允许的最大曲线弯曲幅度 px，越小越接近直线')
    line_options.add_argument('--line-draw-kinematic-max-accel', dest='line_draw_kinematic_max_accel_px_per_ms2', type=float, default=0.18,
                              help='P7 允许的最大拟合加速度 px/ms^2，超出则退回直线')
    line_options.add_argument('--line-draw-kinematic-sample-step-px', dest='line_draw_kinematic_sample_step_px', type=float, default=7.0,
                              help='P7 拟合曲线绘制采样间隔 px，越小越平滑')
    line_options.add_argument('--disable-model-track', dest='model_track_enabled', action='store_false', default=True,
                              help='禁用 P8 模型轨迹主导，退回 bbox 中心轨迹')
    line_options.add_argument('--model-soft-residual-px', dest='model_soft_residual_px', type=float, default=9.0,
                              help='P8 模型轨迹正常观测残差阈值 px')
    line_options.add_argument('--model-hard-residual-px', dest='model_hard_residual_px', type=float, default=22.0,
                              help='P8 模型轨迹弱校正残差阈值 px')
    line_options.add_argument('--model-outlier-residual-px', dest='model_outlier_residual_px', type=float, default=32.0,
                              help='P8 超过此残差认为 bbox 中心离群，不再拉动子弹轨迹')
    line_options.add_argument('--model-correction-gain', dest='model_correction_gain', type=float, default=0.35,
                              help='P8 正常观测对模型的校正权重')
    line_options.add_argument('--model-weak-correction-gain', dest='model_weak_correction_gain', type=float, default=0.10,
                              help='P8 可疑观测对模型的弱校正权重')
    line_options.add_argument('--model-max-accel', dest='model_max_accel_px_ms2', type=float, default=0.10,
                              help='P9 模型速度变化上限 px/ms^2，防止单点观测把轨迹拉弯')
    line_options.add_argument('--model-outlier-kill-enabled', dest='model_outlier_kill_enabled', action='store_true', default=True,
                              help='启用 P9 大残差立即终止；JSON 正向开关兼容，默认启用')
    line_options.add_argument('--disable-model-outlier-kill', dest='model_outlier_kill_enabled', action='store_false', default=True,
                              help='禁用 P9 大残差立即终止；默认启用，防止轨迹穿过人物区域')
    line_options.add_argument('--model-outlier-kill-frames', dest='model_outlier_kill_frames', type=int, default=1,
                              help='连续多少帧模型大残差后终止，默认 1')
    line_options.add_argument('--raw-id-sticky-guard-enabled', dest='raw_id_sticky_guard_enabled', action='store_true', default=True,
                              help='P13 启用 raw_id 粘背景保护，防止 Spatter raw_id 停在黑色障碍物上后把子弹轨迹拉断；默认启用')
    line_options.add_argument('--disable-raw-id-sticky-guard', dest='raw_id_sticky_guard_enabled', action='store_false', default=True,
                              help='禁用 P13 raw_id 粘背景保护')
    line_options.add_argument('--raw-id-sticky-guard-min-speed', dest='raw_id_sticky_guard_min_speed_px_ms', type=float, default=0.35,
                              help='P13 raw_id 粘背景保护启用的最低模型速度 px/ms')
    line_options.add_argument('--raw-id-sticky-guard-static-step-px', dest='raw_id_sticky_guard_static_step_px', type=float, default=2.2,
                              help='P13 已确认子弹 raw_id 直连时，低于该步长视为疑似粘背景')
    line_options.add_argument('--raw-id-sticky-guard-residual-px', dest='raw_id_sticky_guard_residual_px', type=float, default=24.0,
                              help='P13 raw_id 直连点距离模型预测超过该值时不再直接绑定旧弹道')
    line_options.add_argument('--bullet-id-stitch-enabled', dest='bullet_id_stitch_enabled', action='store_true', default=True,
                              help='P16 启用已确认子弹段 ID stitching；新 raw_id 若几何上是旧 bullet 的延续，则继承旧 bullet_id/display_id')
    line_options.add_argument('--disable-bullet-id-stitch', dest='bullet_id_stitch_enabled', action='store_false', default=True,
                              help='禁用 P16 bullet_id stitching')
    line_options.add_argument('--bullet-id-stitch-window-ms', dest='bullet_id_stitch_window_ms', type=float, default=120.0,
                              help='P16 已确认 bullet 段续接最大时间窗口 ms')
    line_options.add_argument('--bullet-id-stitch-corridor-px', dest='bullet_id_stitch_corridor_px', type=float, default=24.0,
                              help='P16 新段到旧弹道方向走廊的最大横向距离 px')
    line_options.add_argument('--bullet-id-stitch-min-dir-cos', dest='bullet_id_stitch_min_dir_cos', type=float, default=0.82,
                              help='P16 新段方向和旧弹道方向的最小 cos 值')
    line_options.add_argument('--bullet-id-stitch-min-speed', dest='bullet_id_stitch_min_speed_px_ms', type=float, default=0.32,
                              help='P16 续接所需最小局部/隐含速度 px/ms')
    line_options.add_argument('--bullet-id-stitch-max-size-ratio', dest='bullet_id_stitch_max_size_ratio', type=float, default=5.5,
                              help='P16 续接允许的新旧 bbox 最大尺寸比例')
    line_options.add_argument('--bullet-id-stitch-reverse-enabled', dest='bullet_id_stitch_reverse_enabled', action='store_true', default=True,
                              help='P17 启用反向直线 ID stitching；新段沿自身直线反向查找旧 bullet_id，补偿遮挡/旧 raw_id 粘背景')
    line_options.add_argument('--disable-bullet-id-stitch-reverse', dest='bullet_id_stitch_reverse_enabled', action='store_false', default=True,
                              help='禁用 P17 反向直线 ID stitching')
    line_options.add_argument('--bullet-id-stitch-reverse-window-ms', dest='bullet_id_stitch_reverse_window_ms', type=float, default=170.0,
                              help='P17 反向直线查找旧 bullet_id 的最大时间窗口 ms')
    line_options.add_argument('--bullet-id-stitch-reverse-corridor-px', dest='bullet_id_stitch_reverse_corridor_px', type=float, default=26.0,
                              help='P17 旧 bullet 到新段反向延长线的最大横向距离 px')
    line_options.add_argument('--bullet-id-stitch-reverse-min-dir-cos', dest='bullet_id_stitch_reverse_min_dir_cos', type=float, default=0.82,
                              help='P17 新段方向和旧 bullet 方向的最小 cos 值')
    line_options.add_argument('--bullet-id-stitch-reverse-min-local-disp-px', dest='bullet_id_stitch_reverse_min_local_disp_px', type=float, default=10.0,
                              help='P17 允许反向续接前，新段自身至少需要的局部位移 px')
    line_options.add_argument('--bullet-id-stitch-reverse-max-size-ratio', dest='bullet_id_stitch_reverse_max_size_ratio', type=float, default=5.8,
                              help='P17 反向续接允许的新旧 bbox 最大尺寸比例')
    line_options.add_argument('--bullet-id-bounce-link-enabled', dest='bullet_id_bounce_link_enabled', action='store_true', default=True,
                              help='P19 启用 C++ 物理子弹反弹续接：反弹前后段继承同一 bullet_id/shot，但不直接判 HIT')
    line_options.add_argument('--disable-bullet-id-bounce-link', dest='bullet_id_bounce_link_enabled', action='store_false', default=True,
                              help='禁用 P19 反弹续接')
    line_options.add_argument('--bullet-id-bounce-link-window-ms', dest='bullet_id_bounce_link_window_ms', type=float, default=260.0,
                              help='P19 反弹续接最大时间窗口 ms')
    line_options.add_argument('--bullet-id-bounce-link-max-gap-px', dest='bullet_id_bounce_link_max_gap_px', type=float, default=95.0,
                              help='P19 反弹续接基础最大空间间隔 px')
    line_options.add_argument('--bullet-id-bounce-link-adaptive-max-px', dest='bullet_id_bounce_link_adaptive_max_px', type=float, default=190.0,
                              help='P19 反弹续接速度自适应最大空间间隔 px')
    line_options.add_argument('--bullet-id-bounce-link-min-turn-angle-deg', dest='bullet_id_bounce_link_min_turn_angle_deg', type=float, default=95.0,
                              help='P19 反弹续接最小方向变化角度 deg')
    line_options.add_argument('--bullet-id-bounce-link-min-speed', dest='bullet_id_bounce_link_min_speed_px_ms', type=float, default=0.38,
                              help='P19 反弹续接最低前段/后段/隐含速度 px/ms')
    line_options.add_argument('--bullet-id-bounce-link-max-size-ratio', dest='bullet_id_bounce_link_max_size_ratio', type=float, default=6.5,
                              help='P19 反弹续接允许的新旧 bbox 最大尺寸比例')
    line_options.add_argument('--bullet-id-bounce-link-candidate-max-points', dest='bullet_id_bounce_link_candidate_max_points', type=int, default=12,
                              help='P19 新段最多积累多少点仍允许反弹继承旧 bullet_id')
    line_options.add_argument('--line-draw-kinematic-force-straight-line', dest='line_draw_kinematic_force_straight_line', action='store_true', default=True,
                              help='启用 P9 分段直线绘制；JSON 正向开关兼容，默认启用')
    line_options.add_argument('--disable-line-draw-force-straight', dest='line_draw_kinematic_force_straight_line', action='store_false', default=True,
                              help='禁用 P9 分段直线绘制；默认强制单段直线，不拟合二次曲线')
    line_options.add_argument('--line-draw-after-terminate-hold-ms', dest='line_draw_after_terminate_hold_ms', type=int, default=0,
                              help='终止后额外保留轨迹显示多久；P9 默认 0，即关闭飞行残留')
    line_options.add_argument('--line-draw-extend-tail-after-terminate', dest='line_draw_extend_tail_after_terminate', action='store_true', default=False,
                              help='启用终止后速度外推尾迹；JSON 正向开关兼容，P9 默认关闭')
    line_options.add_argument('--enable-line-draw-extend-tail-after-terminate', dest='line_draw_extend_tail_after_terminate', action='store_true', default=False,
                              help='启用终止后速度外推尾迹；P9 默认关闭')
    line_options.add_argument('--line-bootstrap-frames',         dest='line_bootstrap_frames',         type=int,   default=0)
    line_options.add_argument('--line-angle-thresh-deg',         dest='line_angle_thresh_deg',         type=float, default=16.0)
    line_options.add_argument('--line-max-offset-px',            dest='line_max_offset_px',            type=float, default=4.0)
    line_options.add_argument('--line-min-step-px',              dest='line_min_step_px',              type=float, default=3.0)
    line_options.add_argument('--line-min-total-disp-px',        dest='line_min_total_disp_px',        type=float, default=20.0)
    line_options.add_argument('--line-max-missed-frames',        dest='line_max_missed_frames',        type=int,   default=4)
    line_options.add_argument('--line-same-direction-ratio-thresh', dest='line_same_direction_ratio_thresh', type=float, default=0.75)
    line_options.add_argument('--line-min-valid-step-count',     dest='line_min_valid_step_count',     type=int,   default=2)
    line_options.add_argument('--line-min-valid-step-ratio',     dest='line_min_valid_step_ratio',     type=float, default=0.55)
    line_options.add_argument('--bullet-min-output-streak',      dest='bullet_min_output_streak',      type=int,   default=5)
    line_options.add_argument('--assoc-max-distance-px',         dest='assoc_max_distance_px',         type=float, default=36.0)
    line_options.add_argument('--assoc-direction-penalty-px',    dest='assoc_direction_penalty_px',    type=float, default=10.0)
    line_options.add_argument('--assoc-max-size-ratio',          dest='assoc_max_size_ratio',          type=float, default=3.0)
    # P0/P1 新增参数
    line_options.add_argument('--maintain-max-offset-px',        dest='maintain_max_offset_px',        type=float, default=5.0,
                              help='maintain 阶段 max_offset 超此值立即 kill (default: 5.0)')
    line_options.add_argument('--maintain-max-static-frames',    dest='maintain_max_static_frames',    type=int,   default=5,
                              help='maintain 阶段 static_fail_count 超此值 kill (default: 5, 原18)')
    line_options.add_argument('--maintain-max-bbox-std',         dest='maintain_max_bbox_std',         type=float, default=4.5,
                              help='maintain 阶段 bbox 宽高 std 超此值 kill (default: 4.5)')
    line_options.add_argument('--probation-max-offset-px',       dest='probation_max_offset_px',       type=float, default=4.5,
                              help='probation 阶段 max_offset 上限 (default: 4.5)')
    line_options.add_argument('--disable-ghost-capture',         dest='ghost_capture_enabled',         action='store_false', default=True,
                              help='禁用 ghost 跨 track 捕获')
    # P2-A: recent offset 连续超限快速 kill
    line_options.add_argument('--maintain-recent-offset-px',           dest='maintain_recent_offset_px',           type=float, default=4.0,
                              help='maintain 阶段 recent_max_offset 超此值连续 K 帧则 kill (default: 4.0)')
    line_options.add_argument('--maintain-recent-offset-exceed-frames', dest='maintain_recent_offset_exceed_frames', type=int,   default=3,
                              help='recent_offset 超限连续帧数阈值 (default: 3)')
    # P2-B: bbox EMA 慢漂移检测
    line_options.add_argument('--maintain-bbox-ema-alpha',      dest='maintain_bbox_ema_alpha',      type=float, default=0.25,
                              help='bbox EMA 平滑系数，越小越平滑 (default: 0.25)')
    line_options.add_argument('--maintain-bbox-ema-drift-ratio', dest='maintain_bbox_ema_drift_ratio', type=float, default=0.45,
                              help='当前帧 bbox 相对 EMA 均值的最大允许偏差率 (default: 0.45)')
    # P2-C: maintain_turn 次数和累计角度限制
    line_options.add_argument('--maintain-max-turn-count',          dest='maintain_max_turn_count',          type=int,   default=2,
                              help='允许的最大 maintain_turn 次数，超过则 kill (default: 2)')
    line_options.add_argument('--maintain-max-cumulative-turn-deg', dest='maintain_max_cumulative_turn_deg', type=float, default=90.0,
                              help='允许的最大累计转向角度（度），超过则 kill (default: 90.0)')
    # P2-E: 反弹续接距离
    line_options.add_argument('--ghost-bounce-max-dist-px',     dest='ghost_bounce_max_dist_px',     type=float, default=35.0,
                              help='反弹续接分支：新点到旧终止点的最大允许距离（像素）(default: 35.0，P3缩小)')
    line_options.add_argument('--ghost-bounce-min-speed',       dest='ghost_bounce_min_speed_px_per_ms', type=float, default=0.60,
                              help='反弹续接最低隐含速度 px/ms，过低视为人体静止重现 (default: 0.60)')
    line_options.add_argument('--same-track-bbox-reuse-min-disp-px', dest='same_track_bbox_reuse_min_disp_px', type=float, default=6.0,
                              help='terminated_bbox / terminated_bbox_ema 后允许 same-track 重用前，当前点相对终止点至少要新增的位移 (default: 6.0)')
    line_options.add_argument('--trigger-max-full-offset-ratio', dest='trigger_max_full_offset_ratio', type=float, default=1.5,
                              help='trigger 阶段 full_offset 上限倍率（相对 max_line_offset_px）(default: 1.5)')
    line_options.add_argument('--trigger-max-sign-flip-ratio',  dest='trigger_max_sign_flip_ratio',  type=float, default=0.35,
                              help='trigger 阶段相邻步方向反转比率上限 (default: 0.35)')
    line_options.add_argument('--trigger-min-avg-speed',        dest='trigger_min_avg_speed_px_per_ms', type=float, default=0.60,
                              help='trigger 阶段平均速度下限 px/ms，低于此值视为慢漂移误判 (default: 0.60)')
    line_options.add_argument('--trigger-raw-min-valid-steps',   dest='trigger_raw_min_valid_steps', type=int, default=3,
                              help='raw_id 新生 trigger 的最小 recent valid_step_count (default: 3)')
    line_options.add_argument('--trigger-sparse-burst-max-valid-steps', dest='trigger_sparse_burst_max_valid_steps', type=int, default=2,
                              help='稀疏大跳 veto 的最大 recent valid_step_count (default: 2)')
    line_options.add_argument('--trigger-sparse-burst-min-disp-per-step-px', dest='trigger_sparse_burst_min_disp_per_step_px', type=float, default=18.0,
                              help='稀疏大跳 veto 的最小每步位移 px/step (default: 18.0)')
    line_options.add_argument('--trigger-sparse-burst-min-total-disp-px', dest='trigger_sparse_burst_min_total_disp_px', type=float, default=35.0,
                              help='稀疏大跳 veto 的最小 recent 总位移 (default: 35.0)')
    line_options.add_argument('--trigger-sparse-burst-max-path-over-net', dest='trigger_sparse_burst_max_path_over_net', type=float, default=1.03,
                              help='稀疏大跳 veto 的最大 recent path_over_net (default: 1.03)')
    line_options.add_argument('--trigger-old-raw-age-ms', dest='trigger_old_raw_age_ms', type=float, default=120.0,
                              help='老 raw 轨迹触发更严门槛的年龄阈值 ms (default: 120)')
    line_options.add_argument('--trigger-old-raw-min-valid-steps', dest='trigger_old_raw_min_valid_steps', type=int, default=4,
                              help='老 raw 轨迹 trigger 的最小 recent valid_step_count (default: 4)')
    line_options.add_argument('--probation-new-min-extra-disp-px', dest='probation_new_min_extra_disp_px', type=float, default=8.0,
                              help='new/raw 出生后 probation 期间要求的最小新增位移 (default: 8.0)')
    line_options.add_argument('--probation-new-min-extra-valid-steps', dest='probation_new_min_extra_valid_steps', type=int, default=2,
                              help='new/raw 出生后 probation 期间要求的最小新增有效步数 (default: 2)')
    line_options.add_argument('--debug-draw-raw-clusters',       dest='debug_draw_raw_clusters',       action='store_true', default=False,
                              help='调试用：绘制 SpatterTracker 原始 cluster 框；默认关闭，避免人多时本地窗口绘制过重。')
    line_options.add_argument('--disable-debug-draw-raw-clusters', dest='debug_draw_raw_clusters', action='store_false',
                              help='显式关闭原始 cluster 框绘制。')
    line_options.add_argument('--csvdebug-track',               dest='debug_track_csv',               type=str,   default='')
    line_options.add_argument('--spatter-raw-debug-csv',        dest='spatter_raw_debug_csv',        type=str,   default='',
                              help='诊断用：逐 slice 保存 SpatterTracker 原始 cluster，判断飞溅物层是否先断。')
    line_options.add_argument('--spatter-summary-print-ms',     dest='spatter_summary_print_ms',     type=int,   default=0,
                              help='诊断用：每隔 N ms 打印 Spatter 原始 cluster/accepted 统计；0=关闭。')

    tsf1_pub_options = parser.add_argument_group('TSF1 publish options')
    tsf1_pub_options.add_argument('--enable-tsf1-pub',              dest='enable_tsf1_pub',              default=False, action='store_true')
    tsf1_pub_options.add_argument('--tsf1-pub-name',                dest='tsf1_pub_name',                type=str,   default='bullet_ts_latest')
    tsf1_pub_options.add_argument('--tsf1-pub-scale',               dest='tsf1_pub_scale',               type=float, default=0.7)
    tsf1_pub_options.add_argument('--tsf1-pub-fps',                 dest='tsf1_pub_fps',                 type=float, default=8.0)
    tsf1_pub_options.add_argument('--tsf1-pub-accumulation-time-us',dest='tsf1_pub_accumulation_time_us',type=int,   default=12000)
    tsf1_pub_options.add_argument('--tsf1-pub-alpha',               dest='tsf1_pub_alpha',               type=float, default=1.8)
    tsf1_pub_options.add_argument('--tsf1-pub-beta',                dest='tsf1_pub_beta',                type=float, default=0.0)
    tsf1_pub_options.add_argument('--tsf1-pub-dilate-ksize',        dest='tsf1_pub_dilate_ksize',        type=int,   default=3)
    tsf1_pub_options.add_argument('--tsf1-pub-dilate-iters',        dest='tsf1_pub_dilate_iters',        type=int,   default=1)
    tsf1_pub_options.add_argument('--tsf1-server-ip',               dest='tsf1_server_ip',               type=str,   default='')
    tsf1_pub_options.add_argument('--tsf1-server-port',             dest='tsf1_server_port',             type=int,   default=5001)
    tsf1_pub_options.add_argument('--tsf1-camera-id',               dest='tsf1_camera_id',               type=str,   default='cam0')
    tsf1_pub_options.add_argument('--tsf1-codec',                   dest='tsf1_codec',                   type=str,   default='zlib', choices=['raw', 'zlib'])
    tsf1_pub_options.add_argument('--tsf1-zlib-level',              dest='tsf1_zlib_level',              type=int,   default=1)
    tsf1_pub_options.add_argument('--tsf1-ts-mode',                 dest='tsf1_ts_mode',                 type=str,   default='linear', choices=['linear', 'exp'])
    tsf1_pub_options.add_argument('--tsf1-ts-decay-us',             dest='tsf1_ts_decay_us',             type=int,   default=65000)
    tsf1_pub_options.add_argument('--tsf1-post-alpha',              dest='tsf1_post_alpha',              type=float, default=1.0)
    tsf1_pub_options.add_argument('--tsf1-post-beta',               dest='tsf1_post_beta',               type=float, default=0.0)
    tsf1_pub_options.add_argument('--tsf1-post-dilate-ksize',       dest='tsf1_post_dilate_ksize',       type=int,   default=0)
    tsf1_pub_options.add_argument('--tsf1-post-dilate-iters',       dest='tsf1_post_dilate_iters',       type=int,   default=0)
    tsf1_pub_options.add_argument('--tsf1-auto-start-sender',       dest='tsf1_auto_start_sender',       default=False, action='store_true')
    tsf1_pub_options.add_argument('--tsf1-sender-script',           dest='tsf1_sender_script',           type=str,   default='')
    tsf1_pub_options.add_argument('--enable-event-overlay-pub',      dest='enable_event_overlay_pub',    default=False, action='store_true',
                                  help='发布事件相机黑底渲染画面到共享内存，供 MVS 端做融合预览')
    tsf1_pub_options.add_argument('--event-overlay-pub-name',        dest='event_overlay_pub_name',      type=str,   default='event_overlay_latest')
    tsf1_pub_options.add_argument('--event-overlay-pub-scale',       dest='event_overlay_pub_scale',     type=float, default=1.0)
    tsf1_pub_options.add_argument('--event-overlay-pub-mode',        dest='event_overlay_pub_mode',      type=str,   default='overlay',
                                  choices=['overlay', 'raw'],
                                  help='overlay=事件画面+轨迹；raw=只发布 EventFrameGenerator 原始事件画面')

    obs_options = parser.add_argument_group('OBS/SRT live stream options')
    obs_options.add_argument('--enable-obs-stream', dest='enable_obs_stream', action='store_true', default=False,
                             help='启用 OBS 直播推流：直接把本地 output_img 编码为 SRT 视频流')
    obs_options.add_argument('--obs-srt-port', dest='obs_srt_port', type=int, default=0,
                             help='OBS SRT listener 端口；多相机时每台相机必须不同，如 6101/6102/6103')
    obs_options.add_argument('--obs-fps', dest='obs_fps', type=float, default=25.0,
                             help='OBS 推流帧率上限')
    obs_options.add_argument('--obs-bitrate-kbps', dest='obs_bitrate_kbps', type=int, default=2500,
                             help='OBS 推流视频码率，单位 kbps')
    obs_options.add_argument('--obs-latency-ms', dest='obs_latency_ms', type=int, default=80,
                             help='SRT latency，OBS 端建议填同样数值或更高')
    obs_options.add_argument('--obs-scale', dest='obs_scale', type=float, default=1.0,
                             help='推流画面缩放比例，1.0 表示原始窗口大小；0.5 可显著省 CPU/带宽')
    obs_options.add_argument('--obs-bind-ip', dest='obs_bind_ip', type=str, default='0.0.0.0',
                             help='SRT listener 绑定地址，默认 0.0.0.0')
    obs_options.add_argument('--obs-ffmpeg-bin', dest='obs_ffmpeg_bin', type=str, default='ffmpeg',
                             help='ffmpeg 可执行文件路径')
    obs_options.add_argument('--obs-encoder', dest='obs_encoder', type=str, default='libx264',
                             help='视频编码器，默认 libx264；如确认支持可用 h264_nvenc')
    obs_options.add_argument('--obs-preset', dest='obs_preset', type=str, default='ultrafast',
                             help='libx264 preset，建议 ultrafast 低延迟')
    obs_options.add_argument('--obs-gop', dest='obs_gop', type=int, default=0,
                             help='GOP 长度，0 表示自动等于 obs_fps')
    obs_options.add_argument('--obs-log-file', dest='obs_log_file', type=str, default='',
                             help='ffmpeg 日志文件路径；为空则自动生成 obs_srt_<alias>_<port>.log')
    obs_options.add_argument('--obs-srt-tlpktdrop', dest='obs_srt_tlpktdrop', action='store_true', default=False,
                             help='OBS/SRT 低延迟优先：允许 SRT 丢弃迟到包。默认关闭，优先保证子弹轨迹完整。')
    obs_options.add_argument('--obs-no-srt-tlpktdrop', dest='obs_srt_tlpktdrop', action='store_false',
                             help='OBS/SRT 完整性优先：不主动丢迟到包。')
    obs_options.add_argument('--obs-bullet-persist', dest='obs_bullet_persist', action='store_true', default=False,
                             help='兼容旧配置：OBS 子弹轨迹持久化层，只影响 OBS 输出。')
    obs_options.add_argument('--obs-bullet-persist-decay', dest='obs_bullet_persist_decay', type=float, default=0.90,
                             help='OBS 子弹持久化衰减系数，0.85~0.98 常用。')
    obs_options.add_argument('--obs-bullet-persist-modes', dest='obs_bullet_persist_modes', type=str, default='bullet',
                             help='OBS 子弹持久化作用模式，逗号分隔，例如 bullet 或 bullet,overlay。')
    obs_options.add_argument('--obs-bullet-persist-alpha', dest='obs_bullet_persist_alpha', type=float, default=1.0,
                             help='OBS 子弹持久化层叠加强度，默认 1.0。')
    obs_options.add_argument('--obs-bullet-persist-tau-ms', dest='obs_bullet_persist_tau_ms', type=float, default=100.0,
                             help='兼容旧配置：持久化时间常数 ms；当前优先使用 decay。')
    obs_options.add_argument('--obs-output-mode', dest='obs_output_mode', type=str, default='overlay',
                             choices=['overlay', 'raw', 'bullet'],
                             help=(
                                 'OBS 输出画面模式：'
                                 'overlay=事件画面+子弹轨迹/ID/禁区叠加；'
                                 'raw=只输出事件相机原始渲染画面，不画子弹轨迹/ID；'
                                 'bullet=黑底只输出子弹轨迹/ID。'
                             ))
    obs_options.add_argument('--obs-bullet-only', dest='obs_bullet_only', action='store_true', default=False,
                             help='兼容旧参数：等价于 --obs-output-mode bullet。')
    obs_options.add_argument('--obs-align-by-ts', dest='obs_align_by_ts', action='store_true', default=False,
                             help='v3：OBS 输出按事件相机 sensor timestamp 缓冲对齐。')
    obs_options.add_argument('--obs-no-align-by-ts', dest='obs_align_by_ts', action='store_false',
                             help='关闭 OBS timestamp 对齐，恢复实时最新帧输出。')
    obs_options.add_argument('--obs-align-delay-ms', dest='obs_align_delay_ms', type=int, default=1000,
                             help='v3：OBS timestamp 对齐延迟，单位 ms。两路相机应使用相同值。')
    obs_options.add_argument('--obs-align-buffer-frames', dest='obs_align_buffer_frames', type=int, default=250,
                             help='v3：OBS timestamp 对齐缓冲帧数上限。')

    hit_options = parser.add_argument_group('Bullet hit event options')
    hit_options.add_argument('--enable-hit-events',   dest='enable_hit_events',   default=False, action='store_true',
                             help='启用子弹命中事件 UDP 发送')
    hit_options.add_argument('--hit-server-ip',       dest='hit_server_ip',       type=str,   default='',
                             help='命中判定服务器 IP（通常与 TSF1 服务器相同）')
    hit_options.add_argument('--hit-server-port',     dest='hit_server_port',     type=int,   default=5003,
                             help='命中判定服务器 UDP 端口（默认 5003）')
    hit_options.add_argument('--hit-camera-alias',    dest='hit_camera_alias',    type=str,   default='',
                             help='相机可选显示名，不影响逻辑，--tsf1-camera-id 填序列号即可')

    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument('-f', '--replay_factor', type=float, default=1)

    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument('-o', '--out-video',    dest='out_video', type=str, default="")
    outcome_options.add_argument('--out-video-mode', dest='out_video_mode', type=str, default='overlay',
                                 choices=['overlay', 'raw', 'bullet'],
                                 help=(
                                     '本地保存视频画面模式：'
                                     'overlay=事件画面+子弹轨迹/ID/禁区；'
                                     'raw=只保存事件相机原始渲染画面；'
                                     'bullet=黑底只保存子弹轨迹/ID。'
                                 ))
    outcome_options.add_argument('-l', '--log-results',  dest='out_log',   type=str, default="")
    outcome_options.add_argument('--bullet-effect-log', dest='bullet_effect_log', type=str, default="",
                                 help='标准子弹特效轨迹 CSV。会自动按相机 alias/SN 追加后缀，避免多相机写同一个文件。')

    hw_group = parser.add_argument_group('Live camera HAL tuning')
    hw_group.add_argument('--enable-digital-crop',  action='store_true', default=False)
    hw_group.add_argument('--disable-digital-crop', action='store_true', default=False)
    hw_group.add_argument('--crop-region', nargs=4, type=int, metavar=('START_X', 'START_Y', 'END_X', 'END_Y'), default=None)
    hw_group.add_argument('--crop-reset-origin', action='store_true', default=False)
    hw_group.add_argument('--enable-afk',    action='store_true', default=False)
    hw_group.add_argument('--disable-afk',   action='store_true', default=False)
    hw_group.add_argument('--afk-low-freq',  type=int, default=0)
    hw_group.add_argument('--afk-high-freq', type=int, default=0)
    hw_group.add_argument('--afk-mode', choices=['band_stop', 'band_pass'], default='band_stop')
    hw_group.add_argument('--afk-duty-cycle',       type=float, default=None)
    hw_group.add_argument('--afk-start-threshold',  type=int,   default=None)
    hw_group.add_argument('--afk-stop-threshold',   type=int,   default=None)
    hw_group.add_argument('--enable-hw-trail',  action='store_true', default=False)
    hw_group.add_argument('--disable-hw-trail', action='store_true', default=False)
    hw_group.add_argument('--hw-trail-type', choices=['trail', 'stc_cut_trail', 'stc_keep_trail'], default='stc_cut_trail')
    hw_group.add_argument('--hw-trail-threshold-us', type=int, default=0)
    hw_group.add_argument('--enable-erc',     action='store_true', default=False)
    hw_group.add_argument('--disable-erc',    action='store_true', default=False)
    hw_group.add_argument('--erc-cd-event-rate', type=int, default=None)
    hw_group.add_argument('--set-bias', dest='bias_updates', action='append', default=[], metavar='NAME=VALUE')
    hw_group.add_argument('--print-biases', dest='print_biases', action='store_true', default=False)

    throttle_options = parser.add_argument_group('Adaptive Throttling options')
    throttle_options.add_argument('--max-events-per-slice',      dest='max_events_per_slice',      type=int,   default=30000,
                                  help='Max events per accumulation slice before throttling kicks in.')
    throttle_options.add_argument('--severe-events-per-slice',   dest='severe_events_per_slice',   type=int,   default=60000,
                                  help='Threshold for severe throttling (more aggressive sampling).')
    throttle_options.add_argument('--burst-hold-ms',             dest='burst_hold_ms',             type=int,   default=1500,
                                  help='Time in ms to wait after rate drops before recovering from throttling.')

    debug_options = parser.add_argument_group('Diagnostics options')
    debug_options.add_argument('--print-effective-args',        dest='print_effective_args',        action='store_true')
    debug_options.add_argument('--strict-line-filter-kwargs',   dest='strict_line_filter_kwargs',   action='store_true')
    debug_options.add_argument('--print-event-stats',           dest='print_event_stats',           action='store_true', default=False,
                              help='只打印代码实际收到的事件统计，不改变算法行为。')
    debug_options.add_argument('--event-stats-interval-ms',     dest='event_stats_interval_ms',     type=int, default=500,
                              help='事件统计打印间隔（毫秒）。')
    debug_options.add_argument('--enable-shake-guard',         dest='enable_shake_guard',         action='store_true', default=False,
                              help='启用相机抖动保护：触发时直接跳过重检测流程。')
    debug_options.add_argument('--shake-guard-rate-thresh-mevs', dest='shake_guard_rate_thresh_mevs', type=float, default=14.0,
                              help='raw 事件率阈值（MEv/s），超过则认为进入抖动保护。')
    debug_options.add_argument('--shake-guard-trigger-slices',  dest='shake_guard_trigger_slices',  type=int, default=2,
                              help='连续多少个 slice 超阈值后触发抖动保护。')
    debug_options.add_argument('--shake-guard-hold-ms',         dest='shake_guard_hold_ms',         type=int, default=800,
                              help='触发后至少保持多长时间（毫秒）。')
    debug_options.add_argument('--shake-guard-release-ratio',   dest='shake_guard_release_ratio',   type=float, default=0.85,
                              help='退出阈值比例：rate <= thresh * ratio 才允许退出。')
    debug_options.add_argument('--shake-guard-skip-tsf1',       dest='shake_guard_skip_tsf1',       action='store_true', default=False,
                              help='抖动保护期间同时暂停 TSF1 发布，进一步减负。')
    debug_options.add_argument('--shake-guard-cluster-thresh', dest='shake_guard_cluster_thresh', type=int, default=0,
                              help='cluster 数量保护阈值；>0 时，当前帧 cluster 数超过该值会进入抖动保护，适合验证相机晃动导致 cluster 爆炸的问题。')
    debug_options.add_argument('--disable-spatter-processing', dest='disable_spatter_processing', action='store_true', default=False,
                              help='调试用：跳过飞溅物检测、子弹轨迹过滤和子弹事件发送，只保留 RAW 录制、事件统计、TSF1/画面生成。')
    debug_options.add_argument('--enable-pre-spatter-guard', dest='enable_pre_spatter_guard', action='store_true', default=False,
                              help='启用前置重载保护：在 spatter_tracker.process_events 之前根据事件数/事件率跳过重检测，防止实时窗口积压。')
    debug_options.add_argument('--pre-spatter-max-events-per-slice', dest='pre_spatter_max_events_per_slice', type=int, default=18000,
                              help='前置保护：单个 accumulation slice 的事件数超过该值时进入保护；0 表示不按事件数触发。')
    debug_options.add_argument('--pre-spatter-rate-thresh-mevs', dest='pre_spatter_rate_thresh_mevs', type=float, default=8.0,
                              help='前置保护：raw 事件率超过该值(MEv/s)时进入保护；0 表示不按事件率触发。')
    debug_options.add_argument('--pre-spatter-trigger-slices', dest='pre_spatter_trigger_slices', type=int, default=1,
                              help='前置保护：连续多少个 slice 超阈值后触发，默认 1 表示立即保护。')
    debug_options.add_argument('--pre-spatter-hold-ms', dest='pre_spatter_hold_ms', type=int, default=600,
                              help='前置保护：触发后至少保持多久，期间只刷新轻量画面，不跑 spatter/line_filter。')
    debug_options.add_argument('--pre-spatter-release-ratio', dest='pre_spatter_release_ratio', type=float, default=0.70,
                              help='前置保护：退出阈值比例，事件数和事件率都低于阈值*ratio 后才退出。')
    debug_options.add_argument('--pre-spatter-reset-on-enter', dest='pre_spatter_reset_on_enter', action='store_true', default=True,
                              help='前置保护触发时重建 spatter_tracker/line_filter，清掉被人群/抖动污染的状态；默认启用。')
    debug_options.add_argument('--disable-pre-spatter-reset-on-enter', dest='pre_spatter_reset_on_enter', action='store_false',
                              help='前置保护触发时不重建检测状态，仅跳过重检测。')
    debug_options.add_argument('--pre-spatter-skip-tsf1', dest='pre_spatter_skip_tsf1', action='store_true', default=False,
                              help='前置保护期间也暂停 TSF1 发布，进一步减负；默认仍发布 TSF1。')
    debug_options.add_argument('--pre-spatter-skip-obs', dest='pre_spatter_skip_obs', action='store_true', default=False,
                              help='前置保护期间暂停 OBS 提交，进一步减负；默认仍提交 OBS。')
    debug_options.add_argument('--raw-record-light-mode', dest='raw_record_light_mode', action='store_true', default=False,
                              help='RAW 录制期间自动进入轻量模式：只刷新画面和录制，不跑 spatter/line_filter/CSV，避免录制真实 RAW 时卡顿。')
    debug_options.add_argument('--raw-record-light-skip-obs', dest='raw_record_light_skip_obs', action='store_true', default=True,
                              help='RAW 录制轻量模式下暂停 OBS 提交；默认启用。')
    debug_options.add_argument('--enable-hotkey-raw-record',  dest='enable_hotkey_raw_record',  action='store_true', default=False,
                              help='启用窗口热键 RAW 录制：按 R 开始，再按 R 结束。')
    debug_options.add_argument('--raw-record-dir',              dest='raw_record_dir',              type=str, default='./raw_captures',
                              help='热键录制 RAW 文件保存目录。')
    debug_options.add_argument('--raw-record-prefix',           dest='raw_record_prefix',           type=str, default='capture',
                              help='热键录制 RAW 文件名前缀。')
    debug_options.add_argument('--enable-remote-raw-record-control', dest='enable_remote_raw_record_control', action='store_true', default=False,
                              help='启用 launcher 广播式多相机 RAW 录制控制。')
    debug_options.add_argument('--raw-record-command-file',     dest='raw_record_command_file',     type=str, default='/tmp/eventcam_raw_record_cmd.json',
                              help='多相机广播录制控制命令文件。')
    debug_options.add_argument('--raw-record-command-poll-ms',  dest='raw_record_command_poll_ms',  type=int, default=50,
                              help='轮询广播录制命令文件的间隔（毫秒）。')

    sync_options = parser.add_argument_group('Hardware sync options')
    sync_options.add_argument('--enable-ext-trigger-in', dest='enable_ext_trigger_in',
                              action='store_true', default=False,
                              help='启用事件相机 Trigger In，接收 MVS 帧同步脉冲。')
    sync_options.add_argument('--ext-trigger-channel', dest='ext_trigger_channel',
                              type=str, default='MAIN',
                              help='Trigger In 通道，默认 MAIN。')
    sync_options.add_argument('--ext-trigger-polarity', dest='ext_trigger_polarity',
                              type=int, default=0,
                              help='用于 MVS frame 对齐的 trigger polarity；当前实测建议用 0。')
    sync_options.add_argument('--ext-trigger-log', dest='ext_trigger_log',
                              type=str, default='',
                              help='External Trigger 记录 CSV 路径；为空则不写 CSV。')
    sync_options.add_argument('--ext-trigger-print-every', dest='ext_trigger_print_every',
                              type=int, default=100,
                              help='每收到多少个有效同步 trigger 打印一次；0=不打印。')
    sync_options.add_argument('--ext-trigger-ready-file', dest='ext_trigger_ready_file',
                              type=str, default='',
                              help='事件相机已进入同步记录循环后写出的 ready 标志文件；MVS 可等待此文件后再开始采集。')
    sync_options.add_argument('--ext-trigger-sync-start-file', dest='ext_trigger_sync_start_file',
                              type=str, default='',
                              help='正式同步开始标志文件。设置后，文件出现前 External Trigger 只清空不计有效 sync_index；文件出现后清零并从 0 开始计数。')
    sync_options.add_argument('--ext-trigger-require-sync-start', dest='ext_trigger_require_sync_start',
                              action='store_true', default=False,
                              help='要求等待 --ext-trigger-sync-start-file 后才给 selected trigger 分配 sync_index。')
    sync_options.add_argument('--ext-trigger-sync-start-reset-log', dest='ext_trigger_sync_start_reset_log',
                              action='store_true', default=False,
                              help='sync_start 出现时截断 event_trigger CSV 并重写表头，保证 CSV 中第一个有效 sync_index 从 0 开始。')


    human_mask_options = parser.add_argument_group('Current-frame human mask event prefilter')
    human_mask_options.add_argument('--enable-event-human-mask-filter', dest='event_human_mask_filter_enable',
                                    action='store_true', default=False,
                                    help='启用事件前置人体 mask 过滤：只使用同 sync_index 当前 MVS 帧的人体范围过滤事件相机事件。')
    human_mask_options.add_argument('--disable-event-human-mask-filter', dest='event_human_mask_filter_enable',
                                    action='store_false',
                                    help='关闭事件前置人体 mask 过滤。')
    human_mask_options.add_argument('--event-human-mask-mode', dest='event_human_mask_mode', type=str,
                                    default='hard_events', choices=['hard_events'],
                                    help='人体 mask 过滤模式。hard_events=当前帧人体 mask 内事件直接不进入 spatter/line_filter。')
    human_mask_options.add_argument('--event-human-mask-jsonl-template', dest='event_human_mask_jsonl_template', type=str,
                                    default='./sync_ipc/human_result_{camera}.jsonl',
                                    help='MVS 当前帧人体 JSONL 路径模板，{camera} 会替换为 A/B/C/D/E。')
    human_mask_options.add_argument('--event-human-mask-mvs-config', dest='event_human_mask_mvs_config', type=str,
                                    default='hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json',
                                    help='包含 event->MVS homography 的 MVS 配置文件；代码会取逆矩阵把人体 mask 投到事件相机坐标。')
    human_mask_options.add_argument('--event-human-mask-current-only', dest='event_human_mask_current_only',
                                    action='store_true', default=True,
                                    help='只允许使用当前 sync_index 的人体 mask。默认开启，避免上一帧/附近帧误过滤。')
    human_mask_options.add_argument('--event-human-mask-allow-noncurrent', dest='event_human_mask_current_only',
                                    action='store_false',
                                    help='允许使用最近人体帧；不建议实弹测试使用。')
    human_mask_options.add_argument('--event-human-mask-wait-ms', dest='event_human_mask_wait_ms', type=float, default=0.0,
                                    help='每个 sync_index 首个事件 slice 等待当前帧人体结果的最长时间。0=不等待；若要严格用当前 MVS 帧覆盖 4ms slice，建议 40~60ms。')
    human_mask_options.add_argument('--event-human-mask-frame-hold-ms', dest='event_human_mask_frame_hold_ms', type=float, default=48.0,
                                    help='当前 MVS 帧人体 mask 的事件时间有效窗。事件 slice 只有在同 sync_index 且距离该 trigger 不超过此值时才使用该 mask；0=只按 sync_index，不按时间过期。')
    human_mask_options.add_argument('--event-human-mask-dilate-px', dest='event_human_mask_dilate_px', type=int, default=8,
                                    help='在事件相机坐标系下对人体 mask 膨胀的像素数，覆盖人体边缘抖动。')
    human_mask_options.add_argument('--event-human-mask-erode-px', dest='event_human_mask_erode_px', type=int, default=0,
                                    help='在事件相机坐标系下对人体 mask 腐蚀的像素数；移动误判优先时建议 0。')
    human_mask_options.add_argument('--event-human-mask-min-conf', dest='event_human_mask_min_conf', type=float, default=0.2,
                                    help='低于该 confidence 的人体不参与事件过滤。')
    human_mask_options.add_argument('--event-human-mask-max-records', dest='event_human_mask_max_records', type=int, default=256,
                                    help='每台事件相机缓存多少个 MVS 人体结果帧。')
    human_mask_options.add_argument('--event-human-mask-print-every-ms', dest='event_human_mask_print_every_ms', type=int, default=1000,
                                    help='事件人体 mask 过滤统计打印间隔；0=不打印。')
    human_mask_options.add_argument('--event-human-mask-use-carry-last', dest='event_human_mask_use_carry_last',
                                    action='store_true', default=False,
                                    help='允许 carry_last 人体结果参与事件 hard mask 过滤。默认关闭，因为 carry_last 是旧 mask，容易误删真实子弹事件。')
    human_mask_options.add_argument('--event-human-mask-stats-jsonl', dest='event_human_mask_stats_jsonl', type=str, default='',
                                    help='可选：把每个 event slice 的人体 hard mask 过滤统计写入 JSONL。支持 {camera}。')

    args = parser.parse_args()

    # 兼容旧配置：JSON 里如果还写着 "obs_bullet_only": true，
    # multi_camera_launcher 会生成 --obs-bullet-only，此时强制切到 bullet 模式。
    if getattr(args, 'obs_bullet_only', False):
        args.obs_output_mode = 'bullet'

    if args.process_to and args.process_from > args.process_to:
        print(f"The processing time interval is not valid. [{args.process_from}, {args.process_to}]")
        raise SystemExit(1)
    return args


# ---------------------------------------------------------------------------
# 构造 LinearMotionFilter 的辅助函数（使用新 dataclass 接口）
# ---------------------------------------------------------------------------

def build_line_filter(args) -> LinearMotionFilter:
    """
    从命令行参数构造 LinearMotionFilter（新 dataclass 接口）。
    每个 dataclass 只填对应的参数，职责清晰。
    """
    track = TrackConfig(
        history_len                 = args.line_history_len,
        bootstrap_frames            = args.line_bootstrap_frames,
        max_angle_deg               = args.line_angle_thresh_deg,
        max_line_offset_px          = args.line_max_offset_px,
        min_step_px                 = args.line_min_step_px,
        min_total_displacement_px   = args.line_min_total_disp_px,
        max_missed_frames           = args.line_max_missed_frames,
        same_direction_ratio_thresh = args.line_same_direction_ratio_thresh,
        bullet_min_output_streak    = args.bullet_min_output_streak,
        min_valid_step_count        = args.line_min_valid_step_count,
        min_valid_step_ratio        = args.line_min_valid_step_ratio,
        # P0/P1
        maintain_max_offset_px      = args.maintain_max_offset_px,
        maintain_max_static_frames  = args.maintain_max_static_frames,
        maintain_max_bbox_std       = args.maintain_max_bbox_std,
        probation_max_offset_px     = args.probation_max_offset_px,
        ghost_capture_enabled       = args.ghost_capture_enabled,
        # P2-A
        maintain_recent_offset_px            = args.maintain_recent_offset_px,
        maintain_recent_offset_exceed_frames = args.maintain_recent_offset_exceed_frames,
        # P2-B
        maintain_bbox_ema_alpha       = args.maintain_bbox_ema_alpha,
        maintain_bbox_ema_drift_ratio = args.maintain_bbox_ema_drift_ratio,
        # P2-C
        maintain_max_turn_count          = args.maintain_max_turn_count,
        maintain_max_cumulative_turn_deg = args.maintain_max_cumulative_turn_deg,
        # P2-E / P3-1
        ghost_bounce_max_dist_px = args.ghost_bounce_max_dist_px,
        # P3-2 / P6-1
        ghost_bounce_min_speed_px_per_ms = args.ghost_bounce_min_speed_px_per_ms,
        same_track_bbox_reuse_min_disp_px = args.same_track_bbox_reuse_min_disp_px,
        # P3-3
        trigger_max_full_offset_ratio = args.trigger_max_full_offset_ratio,
        # P3-5
        trigger_max_sign_flip_ratio   = args.trigger_max_sign_flip_ratio,
        # P5-1
        trigger_min_avg_speed_px_per_ms = args.trigger_min_avg_speed_px_per_ms,
        # P7-1
        trigger_raw_min_valid_steps = args.trigger_raw_min_valid_steps,
        trigger_sparse_burst_max_valid_steps = args.trigger_sparse_burst_max_valid_steps,
        trigger_sparse_burst_min_disp_per_step_px = args.trigger_sparse_burst_min_disp_per_step_px,
        trigger_sparse_burst_min_total_disp_px = args.trigger_sparse_burst_min_total_disp_px,
        trigger_sparse_burst_max_path_over_net = args.trigger_sparse_burst_max_path_over_net,
        trigger_old_raw_age_ms = args.trigger_old_raw_age_ms,
        trigger_old_raw_min_valid_steps = args.trigger_old_raw_min_valid_steps,
        # P7-2
        probation_new_min_extra_disp_px = args.probation_new_min_extra_disp_px,
        probation_new_min_extra_valid_steps = args.probation_new_min_extra_valid_steps,
        # P8
        model_track_enabled = args.model_track_enabled,
        model_soft_residual_px = args.model_soft_residual_px,
        model_hard_residual_px = args.model_hard_residual_px,
        model_outlier_residual_px = args.model_outlier_residual_px,
        model_correction_gain = args.model_correction_gain,
        model_weak_correction_gain = args.model_weak_correction_gain,
        model_max_accel_px_ms2 = args.model_max_accel_px_ms2,
        model_outlier_kill_enabled = args.model_outlier_kill_enabled,
        model_outlier_kill_frames = args.model_outlier_kill_frames,
        # P13 raw_id 粘背景保护
        raw_id_sticky_guard_enabled = args.raw_id_sticky_guard_enabled,
        raw_id_sticky_guard_min_speed_px_ms = args.raw_id_sticky_guard_min_speed_px_ms,
        raw_id_sticky_guard_static_step_px = args.raw_id_sticky_guard_static_step_px,
        raw_id_sticky_guard_residual_px = args.raw_id_sticky_guard_residual_px,
        bullet_id_stitch_enabled = args.bullet_id_stitch_enabled,
        bullet_id_stitch_window_ms = args.bullet_id_stitch_window_ms,
        bullet_id_stitch_corridor_px = args.bullet_id_stitch_corridor_px,
        bullet_id_stitch_min_dir_cos = args.bullet_id_stitch_min_dir_cos,
        bullet_id_stitch_min_speed_px_ms = args.bullet_id_stitch_min_speed_px_ms,
        bullet_id_stitch_max_size_ratio = args.bullet_id_stitch_max_size_ratio,
        bullet_id_stitch_reverse_enabled = args.bullet_id_stitch_reverse_enabled,
        bullet_id_stitch_reverse_window_ms = args.bullet_id_stitch_reverse_window_ms,
        bullet_id_stitch_reverse_corridor_px = args.bullet_id_stitch_reverse_corridor_px,
        bullet_id_stitch_reverse_min_dir_cos = args.bullet_id_stitch_reverse_min_dir_cos,
        bullet_id_stitch_reverse_min_local_disp_px = args.bullet_id_stitch_reverse_min_local_disp_px,
        bullet_id_stitch_reverse_max_size_ratio = args.bullet_id_stitch_reverse_max_size_ratio,
        bullet_id_bounce_link_enabled = args.bullet_id_bounce_link_enabled,
        bullet_id_bounce_link_window_ms = args.bullet_id_bounce_link_window_ms,
        bullet_id_bounce_link_max_gap_px = args.bullet_id_bounce_link_max_gap_px,
        bullet_id_bounce_link_adaptive_max_px = args.bullet_id_bounce_link_adaptive_max_px,
        bullet_id_bounce_link_min_turn_angle_deg = args.bullet_id_bounce_link_min_turn_angle_deg,
        bullet_id_bounce_link_min_speed_px_ms = args.bullet_id_bounce_link_min_speed_px_ms,
        bullet_id_bounce_link_max_size_ratio = args.bullet_id_bounce_link_max_size_ratio,
        bullet_id_bounce_link_candidate_max_points = args.bullet_id_bounce_link_candidate_max_points,
    )
    assoc = AssociationConfig(
        max_distance_px      = args.assoc_max_distance_px,
        direction_penalty_px = args.assoc_direction_penalty_px,
        max_size_ratio       = args.assoc_max_size_ratio,
    )
    draw = DrawConfig(
        draw_trajectory         = True,
        history_len             = args.line_draw_history_len,
        pre_confirm_history_len = args.line_pre_confirm_history_len,
        hold_missed_frames      = args.line_draw_hold_missed_frames,
        connect_max_gap_px      = args.line_draw_connect_max_gap_px,
        kinematic_fit_enabled  = args.line_draw_kinematic_fit_enabled,
        kinematic_force_straight_line = args.line_draw_kinematic_force_straight_line,
        kinematic_residual_px  = args.line_draw_kinematic_residual_px,
        kinematic_max_curve_px = args.line_draw_kinematic_max_curve_px,
        kinematic_max_accel_px_per_ms2 = args.line_draw_kinematic_max_accel_px_per_ms2,
        kinematic_sample_step_px = args.line_draw_kinematic_sample_step_px,
        draw_after_terminate_hold_ms = args.line_draw_after_terminate_hold_ms,
        extend_tail_after_terminate = args.line_draw_extend_tail_after_terminate,
    )
    if getattr(args, 'use_cpp_line_filter', False):
        try:
            from cpp_line_motion_filter import CppLinearMotionFilter
        except Exception as exc:
            raise RuntimeError(
                "已设置 --use-cpp-line-filter，但 cpp_line_motion_filter / _cpp_bullet_full_port_stage06 导入失败。"
                "请先在项目根目录运行 ./BUILD_CPP_SHOT_BOUNCE_LINK.sh。"
                f" 原始错误: {exc!r}"
            ) from exc
        print("[LineFilter] using HANDWRITTEN FULL C++ CppLinearMotionFilter FINAL v1")
        return CppLinearMotionFilter(track=track, assoc=assoc, draw=draw)

    print("[LineFilter] using original Python LinearMotionFilter")
    return LinearMotionFilter(track=track, assoc=assoc, draw=draw)


# ---------------------------------------------------------------------------
# 其余辅助函数（与原版完全一致）
# ---------------------------------------------------------------------------

def draw_no_track_zones(img, input_nozone_rois):
    for roi in input_nozone_rois:
        center_x, center_y, radius, inside = roi
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 255) if inside else (0, 255, 0), 1)


def build_spatter_tracker_kwargs(args, width, height):
    return dict(
        width=width, height=height,
        cell_width=args.cell_width, cell_height=args.cell_height,
        untracked_threshold=args.untracked_ths,
        activation_threshold=args.activation_ths,
        apply_filter=args.apply_filter,
        max_distance=args.max_distance,
        min_size=args.min_size, max_size=args.max_size,
        min_track_time=args.min_track_time,
        static_memory_us=args.static_memory_us,
        max_size_variation=args.max_size_variation,
        min_dist_moving_obj_pxl=args.min_dist_moving_obj,
        filter_type=ev_filter_type_dict[args.filtered_polarity_str],
    )


def _format_hal_status_value(v):
    if isinstance(v, list):
        return '[]' if not v else '[' + ', '.join(str(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{}' if not v else '{' + ', '.join(f'{k}={vv}' for k, vv in v.items()) + '}'
    return str(v)


def print_effective_args(args, runtime_switches, spatter_tracker_kwargs, hal_status):
    print("\n========== [PARSED ARGS] ==========")
    for k, v in sorted(vars(args).items()):
        print(f"{k} = {v}")
    print("\n========== [RUNTIME SWITCHES] ==========")
    for k, v in runtime_switches.items():
        print(f"{k} = {v}")
    print("\n========== [SPATTER TRACKER KWARGS USED] ==========")
    for k, v in spatter_tracker_kwargs.items():
        print(f"{k} = {v}")
    print("\n========== [LIVE HAL STATUS] ==========")
    for k, v in hal_status.items():
        print(f"{k} = {_format_hal_status_value(v)}")


def _new_hal_status(args):
    return {
        'live_mode': None,
        'digital_crop_requested': bool(args.enable_digital_crop or args.disable_digital_crop or args.crop_region is not None),
        'digital_crop_applied': False, 'digital_crop_enabled': None, 'digital_crop_region': None, 'digital_crop_error': '',
        'afk_requested': bool(args.enable_afk or args.disable_afk),
        'afk_applied': False, 'afk_enabled': None, 'afk_error': '',
        'hw_trail_requested': bool(args.enable_hw_trail or args.disable_hw_trail),
        'hw_trail_applied': False, 'hw_trail_enabled': None, 'hw_trail_error': '',
        'erc_requested': bool(args.enable_erc or args.disable_erc or args.erc_cd_event_rate is not None),
        'erc_applied': False, 'erc_enabled': None, 'erc_error': '',
        'bias_print_requested': bool(args.print_biases), 'bias_print_done': False,
        'bias_updates_requested': list(args.bias_updates),
        'bias_updates_applied': [], 'bias_updates_failed': [], 'bias_error': '',
    }


def _enum_value_from_names(enum_obj, names):
    for name in names:
        if hasattr(enum_obj, name):
            return getattr(enum_obj, name)
    return None


def _resolve_enum(holder, enum_attr_candidates, member_names):
    for enum_attr in enum_attr_candidates:
        enum_obj = getattr(holder, enum_attr, None)
        if enum_obj is not None:
            value = _enum_value_from_names(enum_obj, member_names)
            if value is not None:
                return value
    return _enum_value_from_names(holder, member_names)


def _apply_digital_crop(device, args, hal_status):
    crop = getattr(device, 'get_i_digital_crop', lambda: None)()
    if crop is None:
        hal_status['digital_crop_error'] = 'facility unavailable'
        print('[HAL] Digital Crop facility unavailable on this device.')
        return
    if args.disable_digital_crop:
        try:
            ok = crop.enable(False)
            hal_status.update({'digital_crop_applied': True, 'digital_crop_enabled': False})
            print(f'[HAL] Digital Crop disabled: {ok}')
        except Exception as exc:
            hal_status['digital_crop_error'] = str(exc)
        return
    if not args.enable_digital_crop:
        return
    if args.crop_region is None:
        hal_status['digital_crop_error'] = '--crop-region missing'
        print('[HAL] Digital Crop requested but --crop-region not provided. Skip.')
        return
    region = tuple(int(v) for v in args.crop_region)
    try:
        ok_set = crop.set_window_region(region, bool(args.crop_reset_origin))
        ok_en  = crop.enable(True)
        hal_status.update({'digital_crop_applied': True, 'digital_crop_enabled': True, 'digital_crop_region': region})
        print(f'[HAL] Digital Crop set={ok_set}, enabled={ok_en}, region={region}')
    except Exception as exc:
        hal_status['digital_crop_error'] = str(exc)


def _apply_antiflicker(device, args, hal_status):
    afk = getattr(device, 'get_i_antiflicker_module', lambda: None)()
    if afk is None:
        hal_status['afk_error'] = 'facility unavailable'
        return
    if args.disable_afk:
        try:
            afk.enable(False)
            hal_status.update({'afk_applied': True, 'afk_enabled': False})
        except Exception as exc:
            hal_status['afk_error'] = str(exc)
        return
    if not args.enable_afk:
        return
    try:
        if args.afk_low_freq > 0 or args.afk_high_freq > 0:
            afk.set_frequency_band(args.afk_low_freq, args.afk_high_freq)
        mode_val = _resolve_enum(afk, ['FilteringMode', 'Mode'], [args.afk_mode.upper(), args.afk_mode])
        if mode_val is not None:
            afk.set_filtering_mode(mode_val)
        if args.afk_duty_cycle is not None:
            try:
                afk.set_duty_cycle(args.afk_duty_cycle)
            except Exception:
                pass
        if args.afk_start_threshold is not None:
            try:
                afk.set_start_threshold(args.afk_start_threshold)
            except Exception:
                pass
        if args.afk_stop_threshold is not None:
            try:
                afk.set_stop_threshold(args.afk_stop_threshold)
            except Exception:
                pass
        afk.enable(True)
        hal_status.update({'afk_applied': True, 'afk_enabled': True})
        print(f'[HAL] AFK enabled, low={args.afk_low_freq}, high={args.afk_high_freq}, mode={args.afk_mode}')
    except Exception as exc:
        hal_status['afk_error'] = str(exc)
        print(f'[HAL] AFK apply failed: {exc}')


def _apply_event_trail(device, args, hal_status):
    trail = getattr(device, 'get_i_event_trail_filter_module', lambda: None)()
    if trail is None:
        hal_status['hw_trail_error'] = 'facility unavailable'
        return
    if args.disable_hw_trail:
        try:
            trail.enable(False)
            hal_status.update({'hw_trail_applied': True, 'hw_trail_enabled': False})
        except Exception as exc:
            hal_status['hw_trail_error'] = str(exc)
        return
    if not args.enable_hw_trail:
        return
    try:
        type_val = _resolve_enum(trail, ['Type', 'FilterType'], [args.hw_trail_type.upper(), args.hw_trail_type])
        if type_val is not None:
            trail.set_type(type_val)
        if args.hw_trail_threshold_us > 0:
            trail.set_threshold(args.hw_trail_threshold_us)
        trail.enable(True)
        hal_status.update({'hw_trail_applied': True, 'hw_trail_enabled': True})
        print(f'[HAL] Event Trail enabled, type={args.hw_trail_type}, threshold={args.hw_trail_threshold_us}us')
    except Exception as exc:
        hal_status['hw_trail_error'] = str(exc)
        print(f'[HAL] Event Trail apply failed: {exc}')


def _apply_erc(device, args, hal_status):
    erc = getattr(device, 'get_i_erc_module', lambda: None)()
    if erc is None:
        hal_status['erc_error'] = 'facility unavailable'
        return
    if args.disable_erc:
        try:
            erc.enable(False)
            hal_status.update({'erc_applied': True, 'erc_enabled': False})
        except Exception as exc:
            hal_status['erc_error'] = str(exc)
        return
    if not args.enable_erc:
        return
    try:
        if args.erc_cd_event_rate is not None:
            erc.set_cd_event_rate(args.erc_cd_event_rate)
        erc.enable(True)
        hal_status.update({'erc_applied': True, 'erc_enabled': True})
        print(f'[HAL] ERC enabled, cd_event_rate={args.erc_cd_event_rate}')
    except Exception as exc:
        hal_status['erc_error'] = str(exc)
        print(f'[HAL] ERC apply failed: {exc}')


def _apply_biases(device, args, hal_status):
    if not args.bias_updates:
        return
    ll_biases = getattr(device, 'get_i_ll_biases', lambda: None)()
    if ll_biases is None:
        hal_status['bias_error'] = 'facility unavailable'
        return
    if args.print_biases:
        try:
            all_b = ll_biases.get_all_biases()
            print('[HAL] Current biases:')
            for name, val in sorted(all_b.items()):
                print(f'  {name} = {val}')
            hal_status['bias_print_done'] = True
        except Exception as exc:
            print(f'[HAL] Print biases failed: {exc}')
    for entry in args.bias_updates:
        if '=' not in entry:
            continue
        name, val_str = entry.split('=', 1)
        name = name.strip()
        try:
            value = int(val_str.strip())
        except ValueError:
            continue
        try:
            before = ll_biases.get(name)
        except Exception:
            before = None
        try:
            ok = ll_biases.set(name, value)
            try:
                after = ll_biases.get(name)
            except Exception:
                after = None
            hal_status['bias_updates_applied'].append(f'{name}:{before}->{after},ok={ok}')
            print(f'[HAL] Bias set: {name}: {before} -> {after}, ok={ok}')
        except Exception as exc:
            hal_status['bias_updates_failed'].append(f'{name}={value} ({exc})')
            print(f'[HAL] Bias set failed for {name}={value}: {exc}')


def _build_iterator(args, hal_status):
    live = is_live_camera(args.event_file_path)
    hal_status['live_mode'] = live
    if live:
        # 多相机支持：当指定了 tsf1_camera_id（序列号）且未指定 event_file_path 时，
        # 将序列号直接传给 initiate_device，使其按序列号打开指定相机。
        # initiate_device 内部逻辑：path 非空且不是文件时，会将其作为序列号传给
        # DeviceDiscovery.open(path)。
        # 多进程同时打开相机时 HAL 可能存在竞争，因此加入重试机制。
        camera_serial = getattr(args, 'tsf1_camera_id', '') or ''
        open_path = args.event_file_path  # 默认空字符串 → 打开第一台
        if not args.event_file_path and camera_serial and camera_serial != 'cam0':
            open_path = camera_serial     # 传序列号让 initiate_device 按序列号打开

        max_retries = 5
        retry_delay = 3.0  # 秒，给前一台相机留足初始化时间
        device = None
        for attempt in range(1, max_retries + 1):
            try:
                if getattr(args, 'enable_ext_trigger_in', False):
                    from metavision_hal import I_TriggerIn
                    ch_name = str(getattr(args, 'ext_trigger_channel', 'MAIN') or 'MAIN').upper()
                    ch = getattr(I_TriggerIn.Channel, ch_name, I_TriggerIn.Channel.MAIN)
                    device = initiate_device(open_path, use_external_triggers=[ch])
                    hal_status['ext_trigger_in_enabled'] = True
                    hal_status['ext_trigger_channel'] = ch_name
                    print(f"[SYNC] External Trigger In enabled: channel={ch_name}", flush=True)
                else:
                    device = initiate_device(open_path)
                    hal_status['ext_trigger_in_enabled'] = False
                if open_path and open_path != args.event_file_path:
                    print(f"[multi-cam] 已按序列号打开相机: {open_path} (第 {attempt} 次尝试)")
                break
            except OSError as e:
                if attempt < max_retries:
                    print(f"[multi-cam] 打开相机 {open_path} 失败 (第 {attempt}/{max_retries} 次): {e}")
                    print(f"[multi-cam] {retry_delay}s 后重试...")
                    import time as _time
                    _time.sleep(retry_delay)
                else:
                    print(f"[multi-cam] 打开相机 {open_path} 最终失败，已重试 {max_retries} 次")
                    raise
        _apply_digital_crop(device, args, hal_status)
        _apply_antiflicker(device, args, hal_status)
        _apply_event_trail(device, args, hal_status)
        _apply_erc(device, args, hal_status)
        _apply_biases(device, args, hal_status)
        from_device = getattr(EventsIterator, 'from_device')
        kwargs_variants = [
            dict(device=device, start_ts=args.process_from,
                 max_duration=args.process_to - args.process_from if args.process_to else None,
                 delta_t=args.accumulation_time_us),
            dict(device=device, delta_t=args.accumulation_time_us),
            dict(device=device),
        ]
        last_exc = None
        for kwargs in kwargs_variants:
            try:
                return EventsIterator.from_device(**kwargs), True, device
            except TypeError as exc:
                last_exc = exc
        raise last_exc if last_exc is not None else RuntimeError('Failed to build EventsIterator from device.')

    mv_iterator = EventsIterator(
        input_path=args.event_file_path,
        start_ts=args.process_from,
        max_duration=args.process_to - args.process_from if args.process_to else None,
        delta_t=args.accumulation_time_us,
    )
    if args.replay_factor > 0 and not live:
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    return mv_iterator, False, None


def _draw_raw_clusters(img, clusters_np, accepted_clusters):
    # 只跳过“最终确认成子弹”的 raw_id，避免灰色飞溅物框和黄色 b_id 框重叠。
    # display_bullet_id < 0 的 line-filter 候选仍然只是候选，不应该画黄色；
    # 但它仍属于 SpatterTracker 原始飞溅物结果，应该保留灰色 r_xxx 框，方便调 Spatter 参数。
    confirmed_raw_ids = set()
    for c in (accepted_clusters or []):
        try:
            if int(c.get('display_bullet_id', -1)) >= 0:
                confirmed_raw_ids.add(int(c.get('raw_id', c.get('id', -1))))
        except Exception:
            continue
    for cluster in clusters_np:
        raw_id = int(cluster['id'])
        if raw_id in confirmed_raw_ids:
            continue
        x = int(round(float(cluster['x'])))
        y = int(round(float(cluster['y'])))
        w = max(1, int(round(float(cluster['width']))))
        h = max(1, int(round(float(cluster['height']))))
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)
        cv2.putText(img, f"r_{raw_id}", (x, max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


class TimeSurfaceSharedPublisher:
    def __init__(self, width, height, writer, args):
        self.w        = int(width)
        self.h        = int(height)
        self.writer   = writer
        self.args     = args
        self.interval_us  = int(round(1_000_000.0 / max(0.1, float(args.tsf1_pub_fps))))
        self.last_ts      = np.full((self.h, self.w), -1, dtype=np.int64)
        self.next_frame_ts = None
        # 【新增】强制最小累积时间，确保生成的 Time Surface 具有足够的信噪比
        self.min_accumulation_us = int(getattr(args, 'tsf1_pub_accumulation_time_us', 12000))
        self.last_publish_ts = 0

    def update_events(self, evs):
        if evs is None or len(evs) == 0:
            return
        self.last_ts[evs['y'], evs['x']] = evs['t']
        if self.next_frame_ts is None:
            self.next_frame_ts = int(evs['t'][-1])

    def _build_ts(self, t_now_us: int) -> np.ndarray:
        valid = self.last_ts > 0
        img   = np.zeros((self.h, self.w), dtype=np.uint8)
        if not np.any(valid):
            return img
        age = np.zeros((self.h, self.w), dtype=np.int64)
        age[valid] = int(t_now_us) - self.last_ts[valid]

        if self.args.tsf1_ts_mode.lower() == 'linear':
            active = valid & (age >= 0) & (age < int(self.args.tsf1_ts_decay_us))
            if np.any(active):
                val = (1.0 - age[active].astype(np.float32) / float(self.args.tsf1_ts_decay_us)) * 255.0
                img[active] = np.clip(val, 0, 255).astype(np.uint8)
        else:
            active = valid & (age >= 0)
            if np.any(active):
                val = np.exp(-age[active].astype(np.float32) / float(self.args.tsf1_ts_decay_us)) * 255.0
                img[active] = np.clip(val, 0, 255).astype(np.uint8)

        alpha = float(getattr(self.args, 'tsf1_post_alpha', 1.0))
        beta  = float(getattr(self.args, 'tsf1_post_beta', 0.0))
        if abs(alpha - 1.0) > 1e-6 or abs(beta) > 1e-6:
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        ksize = int(getattr(self.args, 'tsf1_post_dilate_ksize', 0))
        iters = int(getattr(self.args, 'tsf1_post_dilate_iters', 0))
        if ksize >= 2 and iters > 0:
            img = cv2.dilate(img, np.ones((ksize, ksize), np.uint8), iterations=iters)
        return img

    def publish_up_to(self, current_ts: int):
        if self.writer is None:
            return
        current_ts = int(current_ts)
        
        # 【新增】强制检查累积时间，防止抖动时高频刷新导致服务器过载且识别不准
        if current_ts - self.last_publish_ts < self.min_accumulation_us:
            return

        if self.next_frame_ts is None:
            self.next_frame_ts = current_ts
        if current_ts < self.next_frame_ts:
            return
        ts_img = self._build_ts(current_ts)
        if self.writer.width != ts_img.shape[1] or self.writer.height != ts_img.shape[0]:
            ts_img = cv2.resize(ts_img, (self.writer.width, self.writer.height), interpolation=cv2.INTER_AREA)
        self.writer.write(ts_img, timestamp_us=current_ts)
        
        # 【新增】更新最后发布时间，作为下一次累积的起点
        self.last_publish_ts = current_ts
        self.next_frame_ts = current_ts + self.interval_us


def _resolve_tsf1_sender_script(explicit_path: str) -> str:
    if explicit_path:
        return explicit_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latest_frame_tsf1_sender.py')


def _get_camera_short_alias(args) -> str:
    alias = str(getattr(args, 'hit_camera_alias', '') or '').strip()
    return alias if alias else str(getattr(args, 'tsf1_camera_id', '') or '').strip()


def _get_window_title(args) -> str:
    return f"一时兴起竞技场{_get_camera_short_alias(args)}"




# ---------------------------------------------------------------------------
# Current-frame human mask event prefilter
# ---------------------------------------------------------------------------

def _safe_json_load(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _event_mask_apply_H(H: np.ndarray, x: float, y: float):
    try:
        v = H @ np.array([float(x), float(y), 1.0], dtype=np.float64)
        if abs(float(v[2])) < 1e-9:
            return float(x), float(y)
        return float(v[0] / v[2]), float(v[1] / v[2])
    except Exception:
        return float(x), float(y)


def _event_mask_homography_for_camera(cfg: dict, cam_id: str) -> np.ndarray:
    overlay_base = cfg.get('event_overlay') if isinstance(cfg.get('event_overlay'), dict) else {}
    per_route = overlay_base.get('per_route') if isinstance(overlay_base.get('per_route'), dict) else {}
    rcfg = dict(overlay_base)
    if isinstance(per_route, dict):
        rcfg.update(per_route.get(str(cam_id), {}) if isinstance(per_route.get(str(cam_id), {}), dict) else {})
    hs = rcfg.get('homographies_event_to_mvs') if isinstance(rcfg.get('homographies_event_to_mvs'), dict) else {}
    active = str(rcfg.get('active_homography', '') or '')
    H = None
    if active and active in hs:
        H = hs.get(active)
    elif 'default' in hs:
        H = hs.get('default')
    elif hs:
        H = next(iter(hs.values()))
    if H is not None:
        try:
            arr = np.array(H, dtype=np.float64)
            if arr.shape == (3, 3):
                return arr
        except Exception:
            pass
    scale = float(rcfg.get('scale', 1.0) or 1.0)
    x = float(rcfg.get('x', 0.0) or 0.0)
    y = float(rcfg.get('y', 0.0) or 0.0)
    return np.array([[scale, 0.0, x], [0.0, scale, y], [0.0, 0.0, 1.0]], dtype=np.float64)


class _JsonlTailerLite:
    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.fp = None
        self.offset = 0
        self.inode = None
        self.last_try = 0.0
        self.last_error = ''

    def poll(self):
        out = []
        if self.fp is None:
            now = time.time()
            if now - self.last_try < 0.05:
                return out
            self.last_try = now
            try:
                st = os.stat(self.path)
                self.fp = open(self.path, 'r', encoding='utf-8', errors='replace')
                self.inode = (st.st_dev, st.st_ino)
                self.offset = 0
            except Exception as exc:
                self.last_error = f'open failed: {exc}'
                return out
        try:
            st = os.stat(self.path)
            key = (st.st_dev, st.st_ino)
            if key != self.inode or st.st_size < self.offset:
                try:
                    self.fp.close()
                except Exception:
                    pass
                self.fp = open(self.path, 'r', encoding='utf-8', errors='replace')
                self.inode = key
                self.offset = 0
            self.fp.seek(self.offset)
            while True:
                line = self.fp.readline()
                if not line:
                    break
                self.offset = self.fp.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
        except Exception as exc:
            self.last_error = f'poll failed: {exc}'
            try:
                if self.fp is not None:
                    self.fp.close()
            except Exception:
                pass
            self.fp = None
            self.offset = 0
        return out


class CurrentFrameHumanMaskEventFilter:
    """
    事件前置人体遮罩过滤器。

    核心原则：只使用同一个 sync_index 的“当前 MVS 帧人体 mask”。
    但事件 slice 是 4ms、MVS 大约 40ms 一帧，所以当前帧 mask 必须在该 MVS 帧时间窗内持久生效，
    不能只作用在某一个 4ms slice 上。

    因此这里使用 frame-persistent current mask：
    - sync_index=N 的人体 mask 只用于 sync_index=N 的事件 slice；
    - 同一个 N 内的多个 4ms slice 复用同一个 mask；
    - 一旦进入 N+1，N 的 mask 立即失效，不拿上一帧/附近帧乱滤；
    - 如果 N 的人体结果尚未写入，可在每个 N 的首个 slice 等待 wait_ms，一旦拿到就覆盖该 N 后续 slice。
    """
    def __init__(self, args, width: int, height: int, camera_alias: str):
        self.enabled = bool(getattr(args, 'event_human_mask_filter_enable', False))
        self.mode = str(getattr(args, 'event_human_mask_mode', 'hard_events') or 'hard_events')
        self.width = int(width)
        self.height = int(height)
        self.camera_alias = str(camera_alias or getattr(args, 'hit_camera_alias', '') or getattr(args, 'tsf1_camera_id', '') or '')
        self.current_only = bool(getattr(args, 'event_human_mask_current_only', True))
        self.wait_ms = max(0.0, float(getattr(args, 'event_human_mask_wait_ms', 0.0) or 0.0))
        self.frame_hold_ms = max(0.0, float(getattr(args, 'event_human_mask_frame_hold_ms', 48.0) or 0.0))
        self.frame_hold_us = int(round(self.frame_hold_ms * 1000.0))
        self.dilate_px = max(0, int(getattr(args, 'event_human_mask_dilate_px', 0) or 0))
        self.erode_px = max(0, int(getattr(args, 'event_human_mask_erode_px', 0) or 0))
        self.min_conf = float(getattr(args, 'event_human_mask_min_conf', 0.0) or 0.0)
        self.max_records = max(8, int(getattr(args, 'event_human_mask_max_records', 256) or 256))
        self.print_every_us = max(0, int(getattr(args, 'event_human_mask_print_every_ms', 1000) or 0) * 1000)
        self.use_carry_last = bool(getattr(args, 'event_human_mask_use_carry_last', False))
        self._last_stats = {
            'enabled': bool(self.enabled),
            'applied': False,
            'reason': 'init',
            'n_in': 0,
            'n_out': 0,
            'n_drop': 0,
            'drop_ratio': 0.0,
        }
        self._stats_fp = None
        stats_tmpl = str(getattr(args, 'event_human_mask_stats_jsonl', '') or '').strip()
        self.stats_jsonl_path = ''
        if stats_tmpl and stats_tmpl.lower() not in ('0', 'false', 'none', 'no', 'off'):
            try:
                self.stats_jsonl_path = stats_tmpl.format(camera=self.camera_alias, cam=self.camera_alias, id=self.camera_alias, alias=self.camera_alias)
            except Exception:
                self.stats_jsonl_path = stats_tmpl
            try:
                stats_dir = os.path.dirname(os.path.abspath(self.stats_jsonl_path))
                if stats_dir:
                    os.makedirs(stats_dir, exist_ok=True)
                self._stats_fp = open(self.stats_jsonl_path, 'a', encoding='utf-8')
            except Exception as exc:
                self._stats_fp = None
                self._last_reason = f'stats_jsonl_open_failed:{exc}'
        self._last_print_ts = -1
        self._total_in = 0
        self._total_out = 0
        self._total_drop = 0
        self._last_reason = 'disabled'
        self._records = {}
        self._mask_cache = {}
        self._latest_sync = None
        self._latest_record = None
        self._waited_syncs = set()
        self._active_sync = None
        self._active_record = None
        self._active_mask_info = None

        tmpl = str(getattr(args, 'event_human_mask_jsonl_template', './sync_ipc/human_result_{camera}.jsonl') or '')
        try:
            self.jsonl_path = tmpl.format(camera=self.camera_alias, cam=self.camera_alias, id=self.camera_alias, alias=self.camera_alias)
        except Exception:
            self.jsonl_path = tmpl
        self.tailer = _JsonlTailerLite(self.jsonl_path) if self.enabled and self.jsonl_path else None

        config_path = str(getattr(args, 'event_human_mask_mvs_config', 'hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json') or '')
        if config_path and not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        self.config_path = config_path
        cfg = _safe_json_load(config_path) if config_path else {}
        self.H_event_to_mvs = _event_mask_homography_for_camera(cfg, self.camera_alias)
        try:
            self.H_mvs_to_event = np.linalg.inv(self.H_event_to_mvs)
        except Exception:
            self.H_mvs_to_event = np.eye(3, dtype=np.float64)
            self._last_reason = 'homography_inverse_failed'

        if self.enabled:
            print(
                f"[HumanMaskFilter][{self.camera_alias}] enabled mode={self.mode} current_only={self.current_only} "
                f"wait_ms={self.wait_ms:.1f} frame_hold_ms={self.frame_hold_ms:.1f} "
                f"dilate={self.dilate_px} erode={self.erode_px} use_carry_last={self.use_carry_last} "
                f"jsonl={self.jsonl_path} mvs_config={self.config_path}",
                flush=True,
            )

    @staticmethod
    def _record_sync_index(rec: dict):
        for key in ('sync_index_hint', 'sync_index', 'mvs_frame_num', 'frame_id_local'):
            try:
                if key in rec and rec.get(key) is not None:
                    return int(rec.get(key))
            except Exception:
                continue
        return None

    def _poll(self):
        if not self.enabled or self.tailer is None:
            return 0
        n = 0
        for rec in self.tailer.poll():
            sync_i = self._record_sync_index(rec)
            if sync_i is None:
                continue
            self._records[int(sync_i)] = rec
            self._latest_sync = int(sync_i)
            self._latest_record = rec
            n += 1
        if len(self._records) > self.max_records:
            keys = sorted(self._records.keys())
            for k in keys[:max(0, len(keys) - self.max_records)]:
                self._records.pop(k, None)
                self._mask_cache.pop(k, None)
        return n

    @staticmethod
    def _record_infer_source(rec: dict) -> str:
        if not isinstance(rec, dict):
            return ''
        return str(rec.get('infer_source', rec.get('human_infer_source', '')) or '').lower()

    def _record_allowed_for_hard_filter(self, rec: dict):
        infer_source = self._record_infer_source(rec)
        if infer_source == 'carry_last' and not bool(self.use_carry_last):
            return False, 'carry_last_not_used_for_hard_filter'
        return True, ''

    def _sync_within_current_frame_window(self, sync_i, sync_event_ts_us, slice_ts_us):
        """Return (ok, reason_suffix) for the same-sync frame-persistent mask validity window."""
        if sync_i is None:
            return False, 'no_sync'
        if self.frame_hold_us <= 0:
            return True, 'no_expire'
        if sync_event_ts_us is None:
            # Without trigger timestamp, keep the same-sync rule but cannot time-expire it.
            return True, 'no_trigger_ts'
        try:
            dt_us = int(slice_ts_us) - int(sync_event_ts_us)
        except Exception:
            return True, 'bad_ts_no_expire'
        if dt_us < -2000:
            return False, f'before_frame_window:dt_us={dt_us}'
        if dt_us > self.frame_hold_us:
            return False, f'current_frame_expired:dt_ms={dt_us/1000.0:.1f}'
        return True, f'frame_window:dt_ms={dt_us/1000.0:.1f}'

    def _get_record_for_sync(self, sync_index, sync_event_ts_us=None, slice_ts_us=0):
        """
        Get the human record for the current MVS sync_index.

        Important: this is not a one-shot match. 4ms event slices inside the same
        40ms MVS frame reuse the same current-frame mask. However, a mask from
        sync_index=N is never used after the event stream has advanced to N+1.
        """
        self._poll()
        if sync_index is None:
            if self.current_only:
                return None, 'no_current_sync_index'
            return self._latest_record, 'latest_no_sync'
        try:
            sync_i = int(sync_index)
        except Exception:
            return None, 'bad_current_sync_index'

        ok_window, window_reason = self._sync_within_current_frame_window(sync_i, sync_event_ts_us, slice_ts_us)
        if not ok_window:
            if self._active_sync == sync_i:
                self._active_sync = None
                self._active_record = None
                self._active_mask_info = None
            return None, window_reason

        # Fast path: same sync_index as the active current MVS frame.
        # This is the persistence layer: one MVS mask covers all 4ms event slices
        # in that same MVS frame window.
        if self._active_sync == sync_i and self._active_record is not None:
            ok_rec, block_reason = self._record_allowed_for_hard_filter(self._active_record)
            if ok_rec:
                return self._active_record, f'current_sync_persist:{window_reason}'
            self._active_sync = None
            self._active_record = None
            self._active_mask_info = None
            return None, f'active_record_blocked:{block_reason}:{window_reason}'

        rec = self._records.get(sync_i)
        if rec is not None:
            ok_rec, block_reason = self._record_allowed_for_hard_filter(rec)
            if not ok_rec:
                return None, f'current_record_blocked:{block_reason}:sync={sync_i}:{window_reason}'
            self._active_sync = sync_i
            self._active_record = rec
            self._active_mask_info = None
            return rec, f'current_sync_match:{window_reason}'

        # Wait only once per sync_index. If the current human result arrives a few
        # milliseconds later, later slices keep polling and will start using it;
        # but we do not block every 4ms slice repeatedly for the same missing frame.
        if self.wait_ms > 0 and sync_i not in self._waited_syncs:
            self._waited_syncs.add(sync_i)
            deadline = time.time() + self.wait_ms / 1000.0
            while time.time() < deadline:
                self._poll()
                rec = self._records.get(sync_i)
                if rec is not None:
                    ok_rec, block_reason = self._record_allowed_for_hard_filter(rec)
                    if not ok_rec:
                        return None, f'current_record_blocked_after_wait:{block_reason}:sync={sync_i}:{window_reason}'
                    self._active_sync = sync_i
                    self._active_record = rec
                    self._active_mask_info = None
                    return rec, f'current_sync_match_after_wait:{window_reason}'
                time.sleep(0.001)

        if self.current_only:
            return None, f'current_mask_missing:sync={sync_i}:{window_reason}'
        if self._latest_record is not None:
            ok_rec, block_reason = self._record_allowed_for_hard_filter(self._latest_record)
            if not ok_rec:
                return None, f'fallback_record_blocked:{block_reason}:target={sync_i}:latest={self._latest_sync}:{window_reason}'
        return self._latest_record, f'fallback_latest:target={sync_i}:latest={self._latest_sync}:{window_reason}'

    def _person_polygons_mvs(self, person: dict):
        polys = person.get('mask_polygons') or person.get('polygons') or person.get('poly') or []
        if isinstance(polys, dict):
            polys = list(polys.values())
        if isinstance(polys, list) and polys:
            for poly in polys:
                if isinstance(poly, list) and len(poly) >= 3:
                    yield poly
            return
        box = person.get('bbox_xyxy') or person.get('bbox') or None
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if x2 > x1 and y2 > y1:
                yield [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _build_mask_for_record(self, rec: dict):
        sync_i = self._record_sync_index(rec)
        if sync_i is not None and sync_i in self._mask_cache:
            return self._mask_cache[sync_i]
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        persons = rec.get('persons') if isinstance(rec, dict) else []
        if not isinstance(persons, list):
            persons = []
        used_persons = 0
        used_polys = 0
        for person in persons:
            if not isinstance(person, dict):
                continue
            try:
                conf = float(person.get('confidence', 1.0) or 0.0)
            except Exception:
                conf = 1.0
            if conf < self.min_conf:
                continue
            person_used = False
            for poly in self._person_polygons_mvs(person):
                pts = []
                for q in poly:
                    if not isinstance(q, (list, tuple)) or len(q) < 2:
                        continue
                    ex, ey = _event_mask_apply_H(self.H_mvs_to_event, float(q[0]), float(q[1]))
                    pts.append([int(round(ex)), int(round(ey))])
                if len(pts) >= 3:
                    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [arr], 255)
                    used_polys += 1
                    person_used = True
            if person_used:
                used_persons += 1
        if self.erode_px > 0:
            k = max(1, int(self.erode_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.erode(mask, kernel, iterations=1)
        if self.dilate_px > 0:
            k = max(1, int(self.dilate_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)
        info = {
            'sync_index': int(sync_i) if sync_i is not None else None,
            'infer_source': self._record_infer_source(rec),
            'human_carry_age_sync': rec.get('human_carry_age_sync') if isinstance(rec, dict) else None,
            'person_count': int(len(persons)),
            'used_persons': int(used_persons),
            'used_polygons': int(used_polys),
            'mask_pixels': int(np.count_nonzero(mask)),
        }
        if sync_i is not None:
            self._mask_cache[int(sync_i)] = (mask, info)
        return mask, info

    def _update_stats(self, slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_out, n_drop, info, reason, applied):
        drop_ratio = float(n_drop) / float(max(1, n_in))
        stats = {
            'enabled': bool(self.enabled),
            'mode': str(self.mode),
            'applied': bool(applied),
            'reason': str(reason),
            'camera': self.camera_alias,
            'slice_ts_us': int(slice_ts_us or 0),
            'current_sync_index': None if current_sync_index is None else int(current_sync_index),
            'current_sync_event_ts_us': None if current_sync_event_ts_us is None else int(current_sync_event_ts_us),
            'n_in': int(n_in),
            'n_out': int(n_out),
            'n_drop': int(n_drop),
            'drop_ratio': round(float(drop_ratio), 6),
            'drop_pct': round(float(drop_ratio) * 100.0, 3),
            'dilate_px': int(self.dilate_px),
            'erode_px': int(self.erode_px),
            'min_conf': float(self.min_conf),
            'frame_hold_ms': float(self.frame_hold_ms),
            'use_carry_last': bool(self.use_carry_last),
        }
        if isinstance(info, dict):
            for key in ('sync_index', 'infer_source', 'human_carry_age_sync', 'person_count', 'used_persons', 'used_polygons', 'mask_pixels'):
                if key in info:
                    stats[key] = info.get(key)
        self._last_stats = stats
        if self._stats_fp is not None:
            try:
                self._stats_fp.write(json.dumps(stats, ensure_ascii=False, separators=(',', ':')) + '\n')
                self._stats_fp.flush()
            except Exception:
                pass

    def get_last_stats(self):
        return dict(getattr(self, '_last_stats', {}) or {})

    def filter_events(self, evs, current_sync_index, slice_ts_us, current_sync_event_ts_us=None):
        if (not self.enabled) or evs is None:
            try:
                n0 = int(len(evs)) if evs is not None else 0
            except Exception:
                n0 = 0
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n0, n0, 0, None, 'disabled', False)
            return evs
        n_in = int(len(evs))
        if n_in <= 0:
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, None, 'empty_events', False)
            return evs
        rec, reason = self._get_record_for_sync(current_sync_index, current_sync_event_ts_us, slice_ts_us)
        if rec is None:
            self._last_reason = reason
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, None, reason, False)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, None, reason)
            return evs
        mask, info = self._build_mask_for_record(rec)
        try:
            if info is not None and info.get('sync_index') is not None:
                self._active_mask_info = info
        except Exception:
            pass
        if mask is None or int(info.get('mask_pixels', 0) or 0) <= 0:
            self._last_reason = 'empty_current_mask'
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info, self._last_reason, False)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, self._last_reason)
            return evs
        try:
            xs = evs['x'].astype(np.int32, copy=False)
            ys = evs['y'].astype(np.int32, copy=False)
        except Exception:
            self._last_reason = 'event_dtype_missing_xy'
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info, self._last_reason, False)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, self._last_reason)
            return evs
        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        inside = np.zeros(n_in, dtype=bool)
        inside[valid] = mask[ys[valid], xs[valid]] > 0
        keep = ~inside
        n_drop = int(np.count_nonzero(inside))
        if n_drop <= 0:
            self._last_reason = reason
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info, reason, False)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, reason)
            return evs
        out = evs[keep]
        n_out = int(len(out))
        self._total_in += n_in
        self._total_out += n_out
        self._total_drop += n_drop
        self._last_reason = reason
        self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_out, n_drop, info, reason, True)
        self._maybe_print(slice_ts_us, n_in, n_out, n_drop, info, reason)
        return out

    def _maybe_print(self, slice_ts_us, n_in, n_out, n_drop, info, reason):
        if self.print_every_us <= 0:
            return
        try:
            ts_i = int(slice_ts_us)
        except Exception:
            ts_i = 0
        if self._last_print_ts >= 0 and ts_i - self._last_print_ts < self.print_every_us:
            return
        self._last_print_ts = ts_i
        sync_s = None if info is None else info.get('sync_index')
        mask_pixels = 0 if info is None else int(info.get('mask_pixels', 0) or 0)
        used_persons = 0 if info is None else int(info.get('used_persons', 0) or 0)
        infer_source = '' if info is None else str(info.get('infer_source', '') or '')
        drop_ratio = (float(n_drop) / float(max(1, n_in))) * 100.0
        print(
            f"[HumanMaskFilter][{self.camera_alias}] sync={sync_s} active_sync={self._active_sync} infer_source={infer_source} reason={reason} "
            f"persons={used_persons} mask_px={mask_pixels} events={n_in}->{n_out} drop={n_drop}({drop_ratio:.1f}%) "
            f"total_drop={self._total_drop}",
            flush=True,
        )

# ---------------------------------------------------------------------------
# External Trigger / MVS 硬件同步辅助函数
# ---------------------------------------------------------------------------

def _get_ext_trigger_events_from_iterator(mv_iterator):
    if hasattr(mv_iterator, 'get_ext_trigger_events'):
        return mv_iterator.get_ext_trigger_events()
    reader = getattr(mv_iterator, 'reader', None)
    if reader is not None and hasattr(reader, 'get_ext_trigger_events'):
        return reader.get_ext_trigger_events()
    return []


def _clear_ext_trigger_events_from_iterator(mv_iterator):
    if hasattr(mv_iterator, 'clear_ext_trigger_events'):
        try:
            mv_iterator.clear_ext_trigger_events()
            return
        except Exception:
            pass
    reader = getattr(mv_iterator, 'reader', None)
    if reader is not None and hasattr(reader, 'clear_ext_trigger_events'):
        try:
            reader.clear_ext_trigger_events()
        except Exception:
            pass


def _ext_trigger_field(ev, name, default=0):
    try:
        return int(ev[name])
    except Exception:
        try:
            return int(getattr(ev, name))
        except Exception:
            return int(default)


def _drain_external_triggers(mv_iterator, args, writer, state, slice_ts_us, wall_time_us):
    # 读取事件相机 External Trigger Events，并按指定 polarity 生成 sync_index。
    # 约定：all_index 记录所有边沿；sync_index 只给 selected polarity 分配，从 0 开始。
    # 后续 MVS frame_index 也从 0 开始，理论上 frame_index=N 对应 sync_index=N。
    if not getattr(args, 'enable_ext_trigger_in', False):
        return 0

    trigs = _get_ext_trigger_events_from_iterator(mv_iterator)
    if len(trigs) <= 0:
        return 0

    if bool(state.get('sync_gate_required', False)) and not bool(state.get('sync_started', False)):
        # 正式 sync_start 之前只清空硬件触发队列，不分配有效 sync_index。
        state['prestart_discarded_edges'] = int(state.get('prestart_discarded_edges', 0) or 0) + int(len(trigs))
        _clear_ext_trigger_events_from_iterator(mv_iterator)
        return 0

    selected_polarity = int(getattr(args, 'ext_trigger_polarity', 0))
    print_every = max(0, int(getattr(args, 'ext_trigger_print_every', 100)))
    written_count = 0

    for tr in trigs:
        t = _ext_trigger_field(tr, 't', 0)
        p = _ext_trigger_field(tr, 'p', -1)
        tid = _ext_trigger_field(tr, 'id', -1)

        key = (int(t), int(p), int(tid))
        if key in state['seen']:
            continue
        state['seen'].add(key)
        if len(state['seen']) > 200000:
            state['seen'].clear()

        all_index = state['all_index']
        state['all_index'] += 1

        sync_index = -1
        if int(p) == selected_polarity:
            sync_index = state['sync_index']
            state['sync_index'] += 1
            state['last_selected_sync_index'] = int(sync_index)
            state['last_selected_event_ts_us'] = int(t)
            if print_every > 0 and state['sync_index'] % print_every == 0:
                print(
                    f"[SYNC] selected trigger sync_index={sync_index} "
                    f"event_ts_us={int(t)} polarity={int(p)} id={int(tid)}",
                    flush=True,
                )

        if writer is not None:
            writer.writerow({
                'all_index': int(all_index),
                'sync_index': int(sync_index),
                'event_ts_us': int(t),
                'polarity': int(p),
                'trigger_id': int(tid),
                'slice_ts_us': int(slice_ts_us),
                'wall_time_us': int(wall_time_us),
            })
            written_count += 1

    _clear_ext_trigger_events_from_iterator(mv_iterator)
    return written_count


def _write_ext_trigger_ready_file(args, camera_alias: str, camera_sn: str):
    # 在事件相机已经打开 Trigger In、日志文件已创建、即将进入主循环时写 ready 文件。
    # MVS 端等待该文件后才 StartGrabbing，可避免每次运行都有不稳定 frame_index 偏移。
    ready_arg = str(getattr(args, 'ext_trigger_ready_file', '') or '').strip()
    if not (getattr(args, 'enable_ext_trigger_in', False) and ready_arg):
        return ''
    path = _build_alias_output_path(ready_arg, camera_alias, camera_sn, default_ext='.flag')
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            'ready': True,
            'camera_alias': str(camera_alias or ''),
            'camera_sn': str(camera_sn or ''),
            'wall_time_us': int(_now_wall_time_us()),
            'selected_polarity': int(getattr(args, 'ext_trigger_polarity', 0)),
        }
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        print(f'[SYNC] event trigger ready file written -> {path}', flush=True)
        return path
    except Exception as exc:
        print(f'[SYNC][WARN] failed to write ready file {path}: {exc}', flush=True)
        return ''



def _format_ext_trigger_sync_start_path(raw_path: str, camera_alias: str, camera_sn: str) -> str:
    raw_path = str(raw_path or '').strip()
    if not raw_path:
        return ''
    alias = str(camera_alias or '').strip()
    serial = str(camera_sn or '').strip()
    try:
        raw_path = raw_path.format(
            camera=alias,
            cam=alias,
            alias=alias,
            id=alias,
            serial=serial,
            camera_sn=serial,
        )
    except Exception:
        pass
    return os.path.abspath(os.path.expanduser(raw_path))


def _activate_ext_trigger_sync_start_if_ready(mv_iterator, args, state, ext_trigger_file, ext_trigger_writer, camera_alias: str) -> bool:
    """Activate formal sync origin when sync_start.flag appears."""
    if not bool(state.get('sync_gate_required', False)):
        return False
    if bool(state.get('sync_started', False)):
        return False
    path = str(state.get('sync_start_path', '') or '')
    if not path or not os.path.exists(path):
        return False

    _clear_ext_trigger_events_from_iterator(mv_iterator)

    state['all_index'] = 0
    state['sync_index'] = 0
    state['last_selected_sync_index'] = None
    state['last_selected_event_ts_us'] = None
    try:
        state['seen'].clear()
    except Exception:
        state['seen'] = set()
    state['sync_started'] = True
    state['sync_start_wall_time_us'] = int(_now_wall_time_us())

    if bool(getattr(args, 'ext_trigger_sync_start_reset_log', False)) and ext_trigger_file is not None and ext_trigger_writer is not None:
        try:
            ext_trigger_file.seek(0)
            ext_trigger_file.truncate(0)
            ext_trigger_writer.writeheader()
            ext_trigger_file.flush()
        except Exception as exc:
            print(f"[SYNC][{camera_alias}][WARN] reset event_trigger CSV on sync_start failed: {exc}", flush=True)

    print(
        f"[SYNC][{camera_alias}] formal sync_start detected -> reset EVENT sync_index=0, path={path}",
        flush=True,
    )
    return True


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
        root = f"{root}_{alias}"

    return root + ext


def _maybe_start_tsf1_sender(args):
    if not args.enable_tsf1_pub or not args.tsf1_auto_start_sender:
        return None
    if not args.tsf1_server_ip:
        raise ValueError('开启 --tsf1-auto-start-sender 时必须提供 --tsf1-server-ip')
    sender_script = _resolve_tsf1_sender_script(args.tsf1_sender_script)
    if not os.path.exists(sender_script):
        raise FileNotFoundError(sender_script)
    cmd = [
        sys.executable, sender_script,
        '--frame-pub-name', args.tsf1_pub_name,
        '--server-ip',      args.tsf1_server_ip,
        '--server-port',    str(args.tsf1_server_port),
        '--camera-id',      args.tsf1_camera_id,
        '--send-fps',       str(args.tsf1_pub_fps),
        '--codec',          args.tsf1_codec,
        '--zlib-level',     str(args.tsf1_zlib_level),
    ]
    print('[TSF1] auto-start sender:', ' '.join(cmd))
    return subprocess.Popen(cmd)


def _obs_submit_frame(obs_streamer, frame, frame_ts_us=None):
    if obs_streamer is None or frame is None:
        return False
    if frame_ts_us is not None and hasattr(obs_streamer, 'submit_at_ts'):
        try:
            return bool(obs_streamer.submit_at_ts(frame, int(frame_ts_us)))
        except Exception:
            pass
    return bool(obs_streamer.submit(frame))


def _publish_shared_bgr_frame(writer, frame_bgr, timestamp_us: int) -> None:
    if writer is None or frame_bgr is None:
        return
    try:
        frame = frame_bgr
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.shape[:2] != (writer.height, writer.width):
            frame = cv2.resize(frame, (writer.width, writer.height), interpolation=cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(frame, timestamp_us=timestamp_us)
    except Exception as exc:
        now = time.time()
        last = float(getattr(writer, '_last_overlay_pub_warn_wall', 0.0))
        if now - last > 5.0:
            setattr(writer, '_last_overlay_pub_warn_wall', now)
            print(f'[EventOverlayPub][WARN] publish failed: {type(exc).__name__}: {exc}', flush=True)


def _submit_obs_frame(obs_streamer, source_img, line_filter, accepted_clusters,
                      obs_output_mode='overlay', raw_event_img=None, obs_bullet_only=False,
                      frame_ts_us=None):
    """
    向 OBS 提交画面。

    obs_output_mode:
      - overlay: 发送当前已叠加后的 source_img（事件画面 + 子弹轨迹/ID/辅助标记）
      - raw:     发送 raw_event_img（只由 EventFrameGenerator 生成，不画子弹轨迹/ID）
      - bullet:  发送黑底子弹轨迹/ID 图层

    兼容旧 OBS 平滑参数：--obs-bullet-persist、--obs-bullet-persist-decay、
    --obs-bullet-persist-modes、--obs-bullet-persist-alpha、--obs-bullet-persist-tau-ms。
    只影响 OBS 输出，不改变检测、line_filter 状态、本地窗口和命中事件。
    """
    if obs_streamer is None:
        return

    mode = 'bullet' if obs_bullet_only else str(obs_output_mode or 'overlay').lower()

    persist_enabled = bool(getattr(obs_streamer, 'bullet_persist_enabled', False))
    persist_modes = str(getattr(obs_streamer, 'bullet_persist_modes', 'bullet') or 'bullet').lower()
    persist_mode_set = {x.strip() for x in persist_modes.split(',') if x.strip()}
    use_persist_here = persist_enabled and (mode in persist_mode_set)

    if mode == 'bullet' or (mode == 'overlay' and use_persist_here):
        bullet_only_img = np.zeros_like(source_img)
        line_filter.draw(bullet_only_img, accepted_clusters)

        if use_persist_here:
            layer = getattr(obs_streamer, '_bullet_persist_layer', None)
            if layer is None or getattr(layer, 'shape', None) != bullet_only_img.shape:
                layer = np.zeros_like(bullet_only_img)

            decay = float(getattr(obs_streamer, 'bullet_persist_decay', 0.90))
            decay = max(0.50, min(0.985, decay))
            alpha = float(getattr(obs_streamer, 'bullet_persist_alpha', 1.0))
            alpha = max(0.0, min(2.0, alpha))

            layer = cv2.addWeighted(layer, decay, np.zeros_like(layer), 0.0, 0.0)
            layer = np.maximum(layer, bullet_only_img)
            obs_streamer._bullet_persist_layer = layer

            if alpha != 1.0:
                send_layer = cv2.addWeighted(layer, alpha, np.zeros_like(layer), 0.0, 0.0)
            else:
                send_layer = layer

            if mode == 'bullet':
                _obs_submit_frame(obs_streamer, send_layer.copy(), frame_ts_us)
                return
            # overlay 模式：把持久化子弹层叠到完整叠加画面上，只补 OBS 端的子弹连续性。
            _obs_submit_frame(obs_streamer, np.maximum(source_img, send_layer).copy(), frame_ts_us)
            return

        if mode == 'bullet':
            _obs_submit_frame(obs_streamer, bullet_only_img, frame_ts_us)
            return

    if mode == 'raw':
        _obs_submit_frame(obs_streamer, raw_event_img if raw_event_img is not None else source_img, frame_ts_us)
        return

    # 默认 overlay：沿用原来的完整叠加画面
    _obs_submit_frame(obs_streamer, source_img, frame_ts_us)

def _now_wall_time_us() -> int:
    return int(time.time_ns() // 1000)


def _wall_time_iso(wall_time_us: int) -> str:
    try:
        return datetime.datetime.fromtimestamp(int(wall_time_us) / 1_000_000.0).astimezone().isoformat(timespec='microseconds')
    except Exception:
        return ''


def _safe_int(v, default=-1) -> int:
    try:
        if v is None or v == '':
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _safe_float(v, default=0.0) -> float:
    try:
        if v is None or v == '':
            return float(default)
        return float(v)
    except Exception:
        return float(default)


BULLET_EFFECT_LOG_FIELDS = [
    'camera_sn', 'camera_alias',
    'timestamp_us',
    'capture_wall_time_us', 'capture_wall_time_iso',
    'estimated_wall_time_us', 'estimated_wall_time_iso',
    'session_start_camera_ts_us', 'session_start_wall_time_us', 'session_start_wall_time_iso',
    # ID 映射：raw_id 是 SpatterTracker 飞溅物 id；stable_track_id 是本脚本关联后的稳定轨迹 id；
    # bullet_id / display_bullet_id 是 LinearMotionFilter 判定出的子弹 id。
    'global_bullet_id', 'bullet_id', 'raw_id', 'stable_track_id', 'display_bullet_id',
    'phase', 'segment_index', 'bullet_active', 'accepted_now', 'just_assigned_bullet',
    'assign_kind', 'birth_assign_kind', 'reuse_source', 'reject_reason',
    'shot_link_type', 'shot_link_reason', 'shot_link_parent_bullet_id',
    'shot_link_parent_segment_index', 'shot_link_gap_ms', 'shot_link_gap_dist_px',
    'shot_link_angle_deg', 'shot_link_score',
    'x', 'y', 'width', 'height', 'cx', 'cy',
    'obs_x', 'obs_y', 'model_x', 'model_y', 'pred_x', 'pred_y',
    'model_residual_px', 'model_obs_used', 'model_outlier', 'model_update_mode',
    'model_speed_px_ms', 'impact_candidate', 'impact_x', 'impact_y',
    'total_disp_px', 'max_offset_px', 'mean_offset_px',
    'direction_ok_ratio', 'same_sign_ratio', 'sign_flip_ratio',
    'valid_step_count', 'valid_step_ratio', 'total_step_count',
    'recent_total_disp_px', 'recent_max_offset_px', 'recent_mean_offset_px',
    'recent_direction_ok_ratio', 'recent_same_sign_ratio', 'recent_sign_flip_ratio',
    'recent_valid_step_count', 'recent_valid_step_ratio',
    'track_age_ms', 'path_over_net', 'recent_path_over_net',
    'keep_streak_after', 'n_points', 'miss_count',
    'turn_count', 'cumulative_turn_deg', 'recent_offset_exceed_count',
    'probation_passed', 'probation_fail_count',
]


def _open_bullet_effect_log(path: str, alias: str, serial: str):
    path = _build_alias_output_path(path, alias, serial, default_ext='.csv')
    if not path:
        return None, None, ''
    log_dir = os.path.dirname(os.path.abspath(path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    f = open(path, mode='w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=BULLET_EFFECT_LOG_FIELDS, extrasaction='ignore')
    writer.writeheader()
    print(f'[BulletEffectLog] enabled -> {path}', flush=True)
    return f, writer, path


def _write_bullet_effect_rows(
    writer, debug_rows, camera_sn: str, camera_alias: str,
    session_start_camera_ts_us: int, session_start_wall_time_us: int,
    capture_wall_time_us: int,
):
    """
    写入标准子弹轨迹日志。

    timestamp_us 是事件相机硬件时间；
    capture_wall_time_us 是当前程序收到/处理这一批数据时的真实 Unix 时间；
    estimated_wall_time_us 用 session 起点把硬件时间换算到真实时间轴，适合和同机可见光日志对齐。
    """
    if writer is None or not debug_rows:
        return
    camera_sn = str(camera_sn or '')
    camera_alias = str(camera_alias or '')
    session_start_camera_ts_us = int(session_start_camera_ts_us or 0)
    session_start_wall_time_us = int(session_start_wall_time_us or capture_wall_time_us or 0)
    for row in debug_rows:
        bullet_id = _safe_int(row.get('bullet_id', -1), -1)
        display_bullet_id = _safe_int(row.get('display_bullet_id', -1), -1)
        bullet_active = _safe_int(row.get('bullet_active', 0), 0)
        # P11/P12: bullet_effect 是正式特效/命中对接日志，只保存已经 display_commit 的轨迹。
        # pending 候选仍保留在 bullet_track_debug 中，避免人体残留进入特效层。
        if display_bullet_id < 0:
            continue

        ts_us = _safe_int(row.get('timestamp', row.get('timestamp_us', 0)), 0)
        estimated_wall_time_us = int(session_start_wall_time_us + (ts_us - session_start_camera_ts_us))
        effect_bid = bullet_id if bullet_id >= 0 else display_bullet_id
        global_bullet_id = f'{camera_sn}_b{effect_bid}' if camera_sn else f'b{effect_bid}'

        writer.writerow({
            'camera_sn': camera_sn,
            'camera_alias': camera_alias,
            'timestamp_us': ts_us,
            'capture_wall_time_us': int(capture_wall_time_us),
            'capture_wall_time_iso': _wall_time_iso(capture_wall_time_us),
            'estimated_wall_time_us': estimated_wall_time_us,
            'estimated_wall_time_iso': _wall_time_iso(estimated_wall_time_us),
            'session_start_camera_ts_us': session_start_camera_ts_us,
            'session_start_wall_time_us': session_start_wall_time_us,
            'session_start_wall_time_iso': _wall_time_iso(session_start_wall_time_us),
            'global_bullet_id': global_bullet_id,
            'bullet_id': bullet_id,
            'raw_id': _safe_int(row.get('raw_id', -1), -1),
            'stable_track_id': _safe_int(row.get('stable_track_id', -1), -1),
            'display_bullet_id': display_bullet_id,
            'phase': str(row.get('phase', '')),
            'segment_index': _safe_int(row.get('segment_index', 0), 0),
            'x': _safe_float(row.get('x', 0.0), 0.0),
            'y': _safe_float(row.get('y', 0.0), 0.0),
            'width': _safe_float(row.get('width', 0.0), 0.0),
            'height': _safe_float(row.get('height', 0.0), 0.0),
            'cx': _safe_float(row.get('cx', 0.0), 0.0),
            'cy': _safe_float(row.get('cy', 0.0), 0.0),
            'total_disp_px': _safe_float(row.get('total_disp_px', 0.0), 0.0),
            'max_offset_px': _safe_float(row.get('max_offset_px', 0.0), 0.0),
            'direction_ok_ratio': _safe_float(row.get('direction_ok_ratio', 0.0), 0.0),
            'same_sign_ratio': _safe_float(row.get('same_sign_ratio', 0.0), 0.0),
            'valid_step_count': _safe_int(row.get('valid_step_count', 0), 0),
            'track_age_ms': _safe_float(row.get('track_age_ms', 0.0), 0.0),
            'path_over_net': _safe_float(row.get('path_over_net', 0.0), 0.0),
            'phase': str(row.get('phase', '')),
            'segment_index': _safe_int(row.get('segment_index', 0), 0),
            'bullet_active': bullet_active,
            'accepted_now': _safe_int(row.get('accepted_now', 0), 0),
            'just_assigned_bullet': _safe_int(row.get('just_assigned_bullet', 0), 0),
            'assign_kind': str(row.get('assign_kind', '')),
            'birth_assign_kind': str(row.get('birth_assign_kind', '')),
            'reuse_source': str(row.get('reuse_source', '')),
            'reject_reason': str(row.get('reject_reason', '')),
            'shot_link_type': str(row.get('shot_link_type', '')),
            'shot_link_reason': str(row.get('shot_link_reason', '')),
            'shot_link_parent_bullet_id': _safe_int(row.get('shot_link_parent_bullet_id', -1), -1),
            'shot_link_parent_segment_index': _safe_int(row.get('shot_link_parent_segment_index', -1), -1),
            'shot_link_gap_ms': _safe_float(row.get('shot_link_gap_ms', 0.0), 0.0),
            'shot_link_gap_dist_px': _safe_float(row.get('shot_link_gap_dist_px', 0.0), 0.0),
            'shot_link_angle_deg': _safe_float(row.get('shot_link_angle_deg', 0.0), 0.0),
            'shot_link_score': _safe_float(row.get('shot_link_score', 0.0), 0.0),
            'mean_offset_px': _safe_float(row.get('mean_offset_px', 0.0), 0.0),
            'sign_flip_ratio': _safe_float(row.get('sign_flip_ratio', 0.0), 0.0),
            'valid_step_ratio': _safe_float(row.get('valid_step_ratio', 0.0), 0.0),
            'total_step_count': _safe_int(row.get('total_step_count', 0), 0),
            'recent_total_disp_px': _safe_float(row.get('recent_total_disp_px', 0.0), 0.0),
            'recent_max_offset_px': _safe_float(row.get('recent_max_offset_px', 0.0), 0.0),
            'recent_mean_offset_px': _safe_float(row.get('recent_mean_offset_px', 0.0), 0.0),
            'recent_direction_ok_ratio': _safe_float(row.get('recent_direction_ok_ratio', 0.0), 0.0),
            'recent_same_sign_ratio': _safe_float(row.get('recent_same_sign_ratio', 0.0), 0.0),
            'recent_sign_flip_ratio': _safe_float(row.get('recent_sign_flip_ratio', 0.0), 0.0),
            'recent_valid_step_count': _safe_int(row.get('recent_valid_step_count', 0), 0),
            'recent_valid_step_ratio': _safe_float(row.get('recent_valid_step_ratio', 0.0), 0.0),
            'recent_path_over_net': _safe_float(row.get('recent_path_over_net', 0.0), 0.0),
            'keep_streak_after': _safe_int(row.get('keep_streak_after', 0), 0),
            'n_points': _safe_int(row.get('n_points', 0), 0),
            'miss_count': _safe_int(row.get('miss_count', 0), 0),
            'turn_count': _safe_int(row.get('turn_count', 0), 0),
            'cumulative_turn_deg': _safe_float(row.get('cumulative_turn_deg', 0.0), 0.0),
            'recent_offset_exceed_count': _safe_int(row.get('recent_offset_exceed_count', 0), 0),
            'probation_passed': _safe_int(row.get('probation_passed', 0), 0),
            'probation_fail_count': _safe_int(row.get('probation_fail_count', 0), 0),
        })



SPATTER_RAW_DEBUG_FIELDS = [
    'timestamp', 'cluster_count', 'accepted_count',
    'raw_id', 'x', 'y', 'width', 'height', 'cx', 'cy', 'area',
    'accepted_now', 'bullet_id', 'display_bullet_id', 'stable_track_id',
    'phase', 'assign_kind', 'reject_reason', 'model_residual_px',
    'miss_count', 'total_disp_px', 'recent_total_disp_px', 'valid_step_count',
]


def _open_spatter_raw_debug_csv(path: str, alias: str, serial: str):
    path = str(path or '').strip()
    if path.lower() in ('', '0', 'false', 'none', 'no', 'off'):
        return None, None, ''
    if path.lower() == 'true':
        path = './bullet_logs/spatter_raw_debug.csv'
    path = _build_alias_output_path(path, alias, serial, default_ext='.csv')
    log_dir = os.path.dirname(os.path.abspath(path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    f = open(path, mode='w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=SPATTER_RAW_DEBUG_FIELDS, extrasaction='ignore')
    writer.writeheader()
    print(f'[SpatterRawDebugLog] enabled -> {path}', flush=True)
    return f, writer, path


def _write_spatter_raw_debug_rows(writer, ts, clusters_np, accepted_clusters, debug_rows):
    """逐 slice 记录 SpatterTracker 的原始输出。关键用途：判断子弹是否在 spatter 层先断。"""
    if writer is None:
        return
    try:
        cluster_count = int(len(clusters_np)) if clusters_np is not None else 0
    except Exception:
        cluster_count = 0
    accepted_count = int(len(accepted_clusters or []))

    row_by_raw = {}
    for row in (debug_rows or []):
        try:
            row_by_raw[int(row.get('raw_id', -999999))] = row
        except Exception:
            pass

    if cluster_count <= 0:
        writer.writerow({
            'timestamp': int(ts), 'cluster_count': 0, 'accepted_count': accepted_count,
            'raw_id': -1, 'accepted_now': 0,
        })
        return

    accepted_raw_ids = set()
    for c in (accepted_clusters or []):
        try:
            accepted_raw_ids.add(int(c['raw_id']))
        except Exception:
            pass

    for c in clusters_np:
        raw_id = _safe_int(c['id'], -1)
        x = _safe_float(c['x'], 0.0)
        y = _safe_float(c['y'], 0.0)
        w = _safe_float(c['width'], 0.0)
        h = _safe_float(c['height'], 0.0)
        drow = row_by_raw.get(raw_id, {})
        writer.writerow({
            'timestamp': int(ts),
            'cluster_count': cluster_count,
            'accepted_count': accepted_count,
            'raw_id': raw_id,
            'x': x, 'y': y, 'width': w, 'height': h,
            'cx': x + w * 0.5, 'cy': y + h * 0.5,
            'area': w * h,
            'accepted_now': 1 if raw_id in accepted_raw_ids else 0,
            'bullet_id': _safe_int(drow.get('bullet_id', -1), -1),
            'display_bullet_id': _safe_int(drow.get('display_bullet_id', -1), -1),
            'stable_track_id': _safe_int(drow.get('stable_track_id', -1), -1),
            'phase': str(drow.get('phase', '')),
            'assign_kind': str(drow.get('assign_kind', '')),
            'reject_reason': str(drow.get('reject_reason', '')),
            'model_residual_px': _safe_float(drow.get('model_residual_px', 0.0), 0.0),
            'miss_count': _safe_int(drow.get('miss_count', 0), 0),
            'total_disp_px': _safe_float(drow.get('total_disp_px', 0.0), 0.0),
            'recent_total_disp_px': _safe_float(drow.get('recent_total_disp_px', 0.0), 0.0),
            'valid_step_count': _safe_int(drow.get('valid_step_count', 0), 0),
        })


def _update_spatter_summary(stats: dict, ts, raw_n, cluster_count, accepted_count, interval_ms: int, camera_log_tag: str, camera_sn_for_log: str):
    if int(interval_ms or 0) <= 0:
        return stats
    if not stats:
        stats = {
            'start_ts': int(ts), 'last_print_ts': int(ts), 'slices': 0,
            'raw_events': 0, 'clusters': 0, 'accepted': 0,
            'zero_cluster_slices': 0, 'max_clusters': 0, 'max_accepted': 0,
        }
    stats['slices'] += 1
    stats['raw_events'] += int(raw_n or 0)
    stats['clusters'] += int(cluster_count or 0)
    stats['accepted'] += int(accepted_count or 0)
    if int(cluster_count or 0) <= 0:
        stats['zero_cluster_slices'] += 1
    stats['max_clusters'] = max(int(stats.get('max_clusters', 0)), int(cluster_count or 0))
    stats['max_accepted'] = max(int(stats.get('max_accepted', 0)), int(accepted_count or 0))

    if int(ts) - int(stats['last_print_ts']) >= int(interval_ms) * 1000:
        slices = max(1, int(stats['slices']))
        print(
            f"[SpatterDiag][{camera_log_tag}][SN={camera_sn_for_log}] "
            f"dt_ms={(int(ts)-int(stats['start_ts']))/1000.0:.1f} "
            f"slices={slices} raw_avg={stats['raw_events']/slices:.1f} "
            f"cluster_avg={stats['clusters']/slices:.2f} cluster_max={stats['max_clusters']} "
            f"zero_slices={stats['zero_cluster_slices']}/{slices} "
            f"accepted_avg={stats['accepted']/slices:.2f} accepted_max={stats['max_accepted']}",
            flush=True,
        )
        stats = {
            'start_ts': int(ts), 'last_print_ts': int(ts), 'slices': 0,
            'raw_events': 0, 'clusters': 0, 'accepted': 0,
            'zero_cluster_slices': 0, 'max_clusters': 0, 'max_accepted': 0,
        }
    return stats

def _maybe_render(ts, output_img, events_frame_gen_algo, line_filter, accepted_clusters,
                  input_nozone_rois, window, video_writer,
                  debug_draw_raw_clusters=False, raw_clusters_np=None, raw_recording=False,
                  obs_streamer=None, submit_obs=True, obs_output_mode='overlay', obs_bullet_only=False,
                  out_video_mode='overlay', event_overlay_pub_writer=None, event_overlay_pub_mode='overlay'):
    events_frame_gen_algo.generate(ts, output_img)

    # 保存一份“纯事件相机渲染画面”：后面即使给本地窗口画子弹轨迹/禁区/REC，
    # OBS raw 模式或 out-video raw 模式仍然可以使用不带子弹轨迹的事件画面。
    out_video_mode = str(out_video_mode or 'overlay').lower()
    event_overlay_pub_mode = str(event_overlay_pub_mode or 'overlay').lower()
    need_raw_img = (
        ((obs_streamer is not None) and (str(obs_output_mode or '').lower() == 'raw'))
        or (video_writer is not None and out_video_mode == 'raw')
        or ((event_overlay_pub_writer is not None) and event_overlay_pub_mode == 'raw')
    )
    raw_event_img = output_img.copy() if need_raw_img else None

    if debug_draw_raw_clusters and raw_clusters_np is not None:
        _draw_raw_clusters(output_img, raw_clusters_np, accepted_clusters)
    line_filter.draw(output_img, accepted_clusters)
    draw_no_track_zones(output_img, input_nozone_rois)
    if raw_recording:
        cv2.putText(output_img, 'REC RAW', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    if window is not None:
        window.show_async(output_img)
    if video_writer is not None:
        if out_video_mode == 'bullet':
            # 黑底视频：只画 LinearMotionFilter 判定出的子弹轨迹/ID，不写入事件背景、禁区和 REC 字样。
            bullet_video_img = np.zeros_like(output_img)
            line_filter.draw(bullet_video_img, accepted_clusters)
            video_writer.write(bullet_video_img)
        elif out_video_mode == 'raw':
            video_writer.write(raw_event_img if raw_event_img is not None else output_img)
        else:
            video_writer.write(output_img)
    if submit_obs:
        _submit_obs_frame(
            obs_streamer, output_img, line_filter, accepted_clusters,
            obs_output_mode=obs_output_mode,
            raw_event_img=raw_event_img,
            obs_bullet_only=obs_bullet_only,
            frame_ts_us=ts,
        )
    if event_overlay_pub_writer is not None:
        if event_overlay_pub_mode == 'raw' and raw_event_img is not None:
            pub_img = raw_event_img
        elif event_overlay_pub_mode in ('bullet', 'bullet_only', 'bullet-only'):
            # Publish only confirmed bullet/line-filter graphics on a black
            # background for fusion preview.  This avoids tinting the MVS live
            # image with the full event-camera frame while keeping bullets clear.
            pub_img = np.zeros_like(output_img)
            line_filter.draw(pub_img, accepted_clusters)
        else:
            pub_img = output_img
        _publish_shared_bgr_frame(event_overlay_pub_writer, pub_img, ts)
    return raw_event_img


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    hal_status = _new_hal_status(args)

    mv_iterator, live, hal_device = _build_iterator(args, hal_status)
    height, width = mv_iterator.get_size()
    print('Camera dimensions:', height, 'x', width)
    print(f'Input mode: {"live camera" if live else "file"}')

    spatter_tracker_kwargs = build_spatter_tracker_kwargs(args, width, height)

    # ---- 新接口：直接从 args 构造三个 dataclass，再传给 LinearMotionFilter ----
    line_filter = build_line_filter(args)

    # ---- 子弹命中事件发送器 ----
    # camera_sn 直接使用 --tsf1-camera-id 的值（建议填相机序列号，如 4110046947）
    bullet_sender = BulletEventSender(
        server_ip  = args.hit_server_ip or args.tsf1_server_ip,
        server_port= args.hit_server_port,
        enabled    = bool(args.enable_hit_events and (args.hit_server_ip or args.tsf1_server_ip)),
    )
    bullet_extractor = BulletEventExtractor(
        sender       = bullet_sender,
        camera_sn    = args.tsf1_camera_id,   # --tsf1-camera-id 填序列号即作为全链路唯一主键
        camera_alias = getattr(args, 'hit_camera_alias', ''),
        orig_w       = width,
        orig_h       = height,
        min_step_px  = args.line_min_step_px,
    )
    bullet_sender.start()

    runtime_switches = {
        'live_mode':       live,
        'stc_enabled':     args.stc_threshold > 0,
        'tsf1_pub_enabled': args.enable_tsf1_pub,
        'event_overlay_pub_enabled': bool(getattr(args, 'enable_event_overlay_pub', False)),
        'replay_wrapped':  args.replay_factor > 0 and not live,
        'headless':        args.headless,
        'render_every_n':  max(1, args.render_every_n),
        'disable_spatter_processing': bool(args.disable_spatter_processing),
        'enable_pre_spatter_guard': bool(getattr(args, 'enable_pre_spatter_guard', False)),
        'pre_spatter_max_events_per_slice': int(getattr(args, 'pre_spatter_max_events_per_slice', 0)),
        'pre_spatter_rate_thresh_mevs': float(getattr(args, 'pre_spatter_rate_thresh_mevs', 0.0)),
        'raw_record_light_mode': bool(getattr(args, 'raw_record_light_mode', False)),
        'obs_output_mode': str(getattr(args, 'obs_output_mode', 'overlay')),
        'shake_guard_cluster_thresh': int(args.shake_guard_cluster_thresh),
        'spatter_raw_debug_csv': str(getattr(args, 'spatter_raw_debug_csv', '') or ''),
        'spatter_summary_print_ms': int(getattr(args, 'spatter_summary_print_ms', 0) or 0),
    }

    if args.print_effective_args:
        print_effective_args(args, runtime_switches, spatter_tracker_kwargs, hal_status)

    spatter_tracker = SpatterTrackerAlgorithm(**spatter_tracker_kwargs)
    clusters        = EventSpatterClusterBuffer()
    
    # 统计模式：保留类定义以兼容旧代码，但这里不启用 throttler，避免测量时改变算法行为。
    throttler = None

    event_stats_last_ts = None
    event_stats_last_wall_us = None
    event_stats_acc_raw = 0
    event_stats_interval_us = max(1, int(args.event_stats_interval_ms) * 1000)

    shake_guard = CameraShakeGuard(
        enabled=args.enable_shake_guard,
        rate_thresh_mevs=args.shake_guard_rate_thresh_mevs,
        trigger_slices=args.shake_guard_trigger_slices,
        hold_ms=args.shake_guard_hold_ms,
        release_ratio=args.shake_guard_release_ratio,
    )
    shake_guard_prev_active = False
    last_loop_ts = None

    camera_short_alias = _get_camera_short_alias(args)
    camera_sn_for_log = str(getattr(args, 'tsf1_camera_id', '') or '').strip()
    camera_log_tag = camera_short_alias if camera_short_alias else camera_sn_for_log if camera_sn_for_log else 'cam'
    window_title = _get_window_title(args)

    cluster_guard_enabled = bool(args.enable_shake_guard and int(args.shake_guard_cluster_thresh) > 0)
    cluster_guard_thresh = max(0, int(args.shake_guard_cluster_thresh))
    cluster_guard_over_count = 0
    cluster_guard_active = False
    cluster_guard_until_ts = -1
    cluster_guard_hold_us = max(0, int(args.shake_guard_hold_ms) * 1000)

    pre_spatter_guard_enabled = bool(getattr(args, 'enable_pre_spatter_guard', False))
    pre_spatter_raw_thresh = max(0, int(getattr(args, 'pre_spatter_max_events_per_slice', 0)))
    pre_spatter_rate_thresh = max(0.0, float(getattr(args, 'pre_spatter_rate_thresh_mevs', 0.0)))
    pre_spatter_trigger_slices = max(1, int(getattr(args, 'pre_spatter_trigger_slices', 1)))
    pre_spatter_hold_us = max(0, int(getattr(args, 'pre_spatter_hold_ms', 600)) * 1000)
    pre_spatter_release_ratio = float(max(0.05, min(1.0, getattr(args, 'pre_spatter_release_ratio', 0.70))))
    pre_spatter_guard_active = False
    pre_spatter_guard_until_ts = -1
    pre_spatter_over_count = 0
    pre_spatter_last_reason = ''
    raw_record_light_prev_active = False

    def _reset_realtime_detection_state(reason: str) -> None:
        # 重建 spatter / cluster buffer / line filter，清掉超载期间产生的污染状态。
        nonlocal spatter_tracker, clusters, line_filter
        spatter_tracker = SpatterTrackerAlgorithm(**spatter_tracker_kwargs)
        clusters = EventSpatterClusterBuffer()
        for roi in input_nozone_rois:
            center_x, center_y, radius, inside = roi
            spatter_tracker.add_nozone(np.array([center_x, center_y]), radius, inside)
        line_filter = build_line_filter(args)
        print(f'[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] reset detection state: {reason}', flush=True)

    if args.disable_spatter_processing:
        print(f'[SpatterDebug][{camera_log_tag}][SN={camera_sn_for_log}] spatter processing disabled; RAW/TSF1/render path only.', flush=True)
    if cluster_guard_enabled:
        print(f'[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] cluster guard enabled: thresh={cluster_guard_thresh}', flush=True)
    if pre_spatter_guard_enabled:
        print(
            f'[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] pre-spatter guard enabled: '
            f'max_events={pre_spatter_raw_thresh}, rate_thresh={pre_spatter_rate_thresh:.2f} MEv/s, '
            f'trigger_slices={pre_spatter_trigger_slices}, hold_ms={int(pre_spatter_hold_us/1000)}',
            flush=True,
        )

    raw_recorder = HotkeyRawRecorder(
        device=hal_device,
        enabled=args.enable_hotkey_raw_record and live,
        output_dir=args.raw_record_dir,
        prefix=args.raw_record_prefix,
        camera_id=camera_short_alias,
    )
    if args.enable_hotkey_raw_record and live:
        print(f'[RAW] hotkey recording enabled. Press R to start/stop. dir={args.raw_record_dir}', flush=True)

    remote_raw_control = RemoteRawRecordController(
        enabled=args.enable_remote_raw_record_control and live,
        command_file=args.raw_record_command_file,
        poll_interval_ms=args.raw_record_command_poll_ms,
    )
    if args.enable_remote_raw_record_control and live:
        print(f'[RAW] remote record control enabled. file={args.raw_record_command_file}', flush=True)
        print('[RAW] press S in any camera window to start/stop synced recording for all cameras.', flush=True)

    raw_record_control_service = AsyncRawRecordControlService(raw_recorder, remote_raw_control)
    if live and (raw_recorder.enabled or remote_raw_control.enabled):
        raw_record_control_service.start()

    do_filtering    = args.stc_threshold > 0
    filtered_evs    = SpatioTemporalContrastAlgorithm.get_empty_output_buffer() if do_filtering else None
    if do_filtering:
        stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_threshold, False)

    event_human_mask_filter = CurrentFrameHumanMaskEventFilter(
        args=args,
        width=width,
        height=height,
        camera_alias=camera_short_alias,
    )

    input_nozone_rois = args.nozone
    for roi in input_nozone_rois:
        center_x, center_y, radius, inside = roi
        spatter_tracker.add_nozone(np.array([center_x, center_y]), radius, inside)
        print(f'No-zone: center=({center_x},{center_y}) radius={radius} inside={inside}')

    events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, args.accumulation_time_us)
    output_img = np.zeros((height, width, 3), np.uint8)
    log = []

    debug_csv_file   = None
    debug_csv_writer = None
    debug_csv_path_arg = str(getattr(args, 'debug_track_csv', '') or '').strip()
    if debug_csv_path_arg.lower() in ('', '0', 'false', 'none', 'no', 'off'):
        debug_csv_path_arg = ''
    elif debug_csv_path_arg.lower() == 'true':
        debug_csv_path_arg = './bullet_logs/bullet_track_debug.csv'

    if debug_csv_path_arg:
        debug_csv_path = _build_alias_output_path(
            debug_csv_path_arg, camera_short_alias, args.tsf1_camera_id, default_ext='.csv'
        )
        debug_csv_dir = os.path.dirname(os.path.abspath(debug_csv_path))
        if debug_csv_dir:
            os.makedirs(debug_csv_dir, exist_ok=True)
        debug_csv_file = open(debug_csv_path, mode='w', newline='', encoding='utf-8')
        debug_csv_writer = csv.DictWriter(debug_csv_file, fieldnames=[
            # 基础 ID：raw_id=飞溅物 id；stable_track_id=关联后的轨迹 id；bullet_id=子弹检测 id
            'timestamp', 'raw_id', 'stable_track_id', 'assign_kind',
            'bullet_id', 'display_bullet_id', 'last_bullet_id',
            'x', 'y', 'width', 'height', 'cx', 'cy',
            'obs_x', 'obs_y', 'model_x', 'model_y', 'pred_x', 'pred_y',
            'model_residual_px', 'model_obs_used', 'model_outlier', 'model_update_mode',
            'model_speed_px_ms', 'impact_candidate', 'impact_x', 'impact_y', 'model_n_points',
            'n_points', 'miss_count', 'keep_now', 'confirmed_now', 'accepted_now',
            'just_assigned_bullet', 'birth_assign_kind',
            'shot_link_type', 'shot_link_reason', 'shot_link_parent_bullet_id',
            'shot_link_parent_segment_index', 'shot_link_parent_stable_track_id',
            'shot_link_parent_x', 'shot_link_parent_y', 'shot_link_gap_ms',
            'shot_link_gap_dist_px', 'shot_link_angle_deg', 'shot_link_score',
            'p12_owner_mode', 'p12_owner_bullet_id', 'p12_inherited_segment_index', 'p12_owner_reason',
            'p12_owner_pending', 'p12_owner_pending_mode', 'p12_polluted_tail',
            'keep_streak_after', 'reject_reason',
            # trigger / maintain 判定状态
            'line_ok', 'direction_ok', 'motion_ok',
            'phase', 'trigger_ok', 'maintain_ok', 'bullet_active',
            'static_fail_count', 'trigger_disp_thresh',
            # 全轨迹几何/运动指标
            'total_disp_px', 'max_offset_px', 'mean_offset_px',
            'direction_ok_ratio', 'same_sign_ratio', 'sign_flip_ratio',
            'valid_step_count', 'valid_step_ratio', 'total_step_count',
            'path_over_net', 'track_age_ms',
            # 最近窗口几何/运动指标
            'recent_total_disp_px', 'recent_max_offset_px', 'recent_mean_offset_px',
            'recent_direction_ok_ratio', 'recent_same_sign_ratio', 'recent_sign_flip_ratio',
            'recent_valid_step_count', 'recent_valid_step_ratio',
            'recent_path_over_net', 'recent_geom_ok', 'recent_motion_ok',
            # 续接/反弹/试用期/转向诊断
            'reuse_source', 'probation_passed', 'probation_fail_count',
            'segment_index', 'compact_ok',
            'turn_count', 'cumulative_turn_deg', 'recent_offset_exceed_count',
        ], extrasaction='ignore')
        debug_csv_writer.writeheader()
        print(f'[BulletDebugLog] enabled -> {debug_csv_path}', flush=True)

    spatter_raw_debug_file = None
    spatter_raw_debug_writer = None
    spatter_raw_debug_file, spatter_raw_debug_writer, _spatter_raw_debug_path = _open_spatter_raw_debug_csv(
        getattr(args, 'spatter_raw_debug_csv', ''), camera_short_alias, args.tsf1_camera_id
    )
    spatter_summary_stats = {}

    bullet_effect_file = None
    bullet_effect_writer = None
    session_start_wall_time_us = _now_wall_time_us()
    session_start_camera_ts_us = None
    if getattr(args, 'bullet_effect_log', ''):
        bullet_effect_file, bullet_effect_writer, _bullet_effect_path = _open_bullet_effect_log(
            args.bullet_effect_log, camera_short_alias, args.tsf1_camera_id
        )

    ext_trigger_file = None
    ext_trigger_writer = None
    ext_trigger_sync_start_path = _format_ext_trigger_sync_start_path(
        str(getattr(args, 'ext_trigger_sync_start_file', '') or ''),
        camera_short_alias,
        args.tsf1_camera_id,
    )
    ext_trigger_gate_required = bool(
        getattr(args, 'enable_ext_trigger_in', False)
        and (bool(getattr(args, 'ext_trigger_require_sync_start', False)) or bool(ext_trigger_sync_start_path))
    )
    ext_trigger_state = {
        'all_index': 0,
        'sync_index': 0,
        'last_selected_sync_index': None,
        'last_selected_event_ts_us': None,
        'seen': set(),
        'sync_gate_required': bool(ext_trigger_gate_required),
        'sync_started': not bool(ext_trigger_gate_required),
        'sync_start_path': ext_trigger_sync_start_path,
        'prestart_discarded_edges': 0,
    }
    ext_trigger_log_arg = str(getattr(args, 'ext_trigger_log', '') or '').strip()
    if getattr(args, 'enable_ext_trigger_in', False) and ext_trigger_log_arg:
        ext_trigger_path = _build_alias_output_path(
            ext_trigger_log_arg, camera_short_alias, args.tsf1_camera_id, default_ext='.csv'
        )
        ext_trigger_dir = os.path.dirname(os.path.abspath(ext_trigger_path))
        if ext_trigger_dir:
            os.makedirs(ext_trigger_dir, exist_ok=True)
        ext_trigger_file = open(ext_trigger_path, mode='w', newline='', encoding='utf-8')
        ext_trigger_writer = csv.DictWriter(ext_trigger_file, fieldnames=[
            'all_index', 'sync_index', 'event_ts_us', 'polarity', 'trigger_id',
            'slice_ts_us', 'wall_time_us',
        ])
        ext_trigger_writer.writeheader()
        print(
            f"[SYNC] external trigger log enabled -> {ext_trigger_path}; "
            f"selected_polarity={int(getattr(args, 'ext_trigger_polarity', 0))}",
            flush=True,
        )
        if ext_trigger_gate_required:
            print(
                f"[SYNC] formal sync_start gate enabled: start_file={ext_trigger_sync_start_path}; "
                f"pre-start trigger edges will be discarded and sync_index will reset to 0",
                flush=True,
            )

    tsf1_pub_writer  = None
    tsf1_sender_proc = None
    if args.enable_tsf1_pub:
        pub_scale = max(0.05, float(args.tsf1_pub_scale))
        pub_w     = max(1, int(round(width  * pub_scale)))
        pub_h     = max(1, int(round(height * pub_scale)))
        tsf1_pub_writer = SharedFrameWriter(
            args.tsf1_pub_name, pub_w, pub_h, channels=1, create=True
        )
        print(f'[TSF1] shared memory enabled: name={args.tsf1_pub_name}, '
              f'size={pub_w}x{pub_h}, max_fps={args.tsf1_pub_fps}')
        tsf1_sender_proc = _maybe_start_tsf1_sender(args)

    tsf1_publisher = (
        TimeSurfaceSharedPublisher(width, height, tsf1_pub_writer, args)
        if args.enable_tsf1_pub else None
    )

    event_overlay_pub_writer = None
    if getattr(args, 'enable_event_overlay_pub', False):
        overlay_scale = max(0.05, float(getattr(args, 'event_overlay_pub_scale', 1.0) or 1.0))
        overlay_w = max(1, int(round(width * overlay_scale)))
        overlay_h = max(1, int(round(height * overlay_scale)))
        event_overlay_pub_writer = SharedFrameWriter(
            str(getattr(args, 'event_overlay_pub_name', 'event_overlay_latest') or 'event_overlay_latest'),
            overlay_w, overlay_h, channels=3, create=True
        )
        print(
            f"[EventOverlayPub] shared memory enabled: name={args.event_overlay_pub_name}, "
            f"size={overlay_w}x{overlay_h}, mode={args.event_overlay_pub_mode}",
            flush=True,
        )

    obs_streamer = None
    if getattr(args, 'enable_obs_stream', False):
        camera_label = _get_camera_short_alias(args)
        if int(getattr(args, 'obs_srt_port', 0) or 0) <= 0:
            print('[OBS-SRT] --enable-obs-stream 已启用，但 --obs-srt-port 未设置或无效，跳过 OBS 推流。', flush=True)
        else:
            obs_streamer = ObsSrtStreamer(
                port=int(args.obs_srt_port),
                enabled=True,
                fps=float(args.obs_fps),
                bitrate_kbps=int(args.obs_bitrate_kbps),
                latency_ms=int(args.obs_latency_ms),
                scale=float(args.obs_scale),
                bind_ip=str(args.obs_bind_ip),
                ffmpeg_bin=str(args.obs_ffmpeg_bin),
                encoder=str(args.obs_encoder),
                preset=str(args.obs_preset),
                gop=int(args.obs_gop),
                label=camera_label,
                log_file=str(args.obs_log_file or ''),
                tlpktdrop=bool(getattr(args, 'obs_srt_tlpktdrop', False)),
                align_by_ts=bool(getattr(args, 'obs_align_by_ts', False)),
                align_delay_ms=int(getattr(args, 'obs_align_delay_ms', 1000)),
                align_buffer_frames=int(getattr(args, 'obs_align_buffer_frames', 250)),
            )
            # OBS bullet persistence compatibility: only affects OBS output, not local detection/window.
            obs_streamer.bullet_persist_enabled = bool(getattr(args, 'obs_bullet_persist', False))
            obs_streamer.bullet_persist_decay = float(getattr(args, 'obs_bullet_persist_decay', 0.90))
            obs_streamer.bullet_persist_modes = str(getattr(args, 'obs_bullet_persist_modes', 'bullet') or 'bullet')
            obs_streamer.bullet_persist_alpha = float(getattr(args, 'obs_bullet_persist_alpha', 1.0))
            obs_streamer.bullet_persist_tau_ms = float(getattr(args, 'obs_bullet_persist_tau_ms', 100.0))
            obs_streamer.start()

    window       = None
    video_writer = None
    video_name   = ''

    try:
        if not args.headless:
            window = MTWindow(
                title=window_title,
                width=width, height=height,
                mode=BaseWindow.RenderMode.BGR,
            )
            window.__enter__()
            window.show_async(output_img)

            def keyboard_cb(key, scancode, action, mods):
                if action != UIAction.RELEASE:
                    return
                if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                    window.set_close_flag()
                    return
                key_r = getattr(UIKeyEvent, 'KEY_R', None)
                if key_r is not None and key == key_r:
                    raw_record_control_service.request_toggle(source='key_r')
                    return
                key_s = getattr(UIKeyEvent, 'KEY_S', None)
                if key_s is not None and key == key_s:
                    if remote_raw_control.enabled:
                        if raw_recorder.is_recording():
                            remote_raw_control.write_command('stop')
                            print('[RAW][SYNC] broadcast stop requested by key S', flush=True)
                        else:
                            session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            remote_raw_control.write_command('start', session_id=session_id)
                            print(f'[RAW][SYNC] broadcast start requested by key S -> session={session_id}', flush=True)
                    else:
                        print('[RAW][SYNC] remote record control not enabled.', flush=True)

            window.set_keyboard_callback(keyboard_cb)

        if args.out_video:
            fourcc     = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = _build_alias_output_path(args.out_video, camera_short_alias, args.tsf1_camera_id, default_ext='.avi')
            video_fps  = max(1.0, 1e6 / (args.accumulation_time_us * max(1, args.render_every_n)))
            video_writer = cv2.VideoWriter(video_name, fourcc, video_fps, (width, height))

        _write_ext_trigger_ready_file(args, camera_short_alias, args.tsf1_camera_id)

        iter_idx = 0
        for evs in mv_iterator:
            ts = mv_iterator.get_current_time()
            if session_start_camera_ts_us is None:
                session_start_camera_ts_us = int(ts)
            capture_wall_time_us = _now_wall_time_us()
            _activate_ext_trigger_sync_start_if_ready(
                mv_iterator=mv_iterator,
                args=args,
                state=ext_trigger_state,
                ext_trigger_file=ext_trigger_file,
                ext_trigger_writer=ext_trigger_writer,
                camera_alias=camera_short_alias,
            )
            ext_trigger_written = _drain_external_triggers(
                mv_iterator=mv_iterator,
                args=args,
                writer=ext_trigger_writer,
                state=ext_trigger_state,
                slice_ts_us=int(ts),
                wall_time_us=int(capture_wall_time_us),
            )
            if ext_trigger_written and ext_trigger_file is not None:
                # 同步链路需要 MVS 进程实时读取 event_trigger_*.csv。
                # 这里必须主动 flush，否则 trigger 行可能长时间停留在 Python 文件缓冲区，
                # MVS 端会误判为“no event_ts_us yet / offset 无法锁定”。
                ext_trigger_file.flush()
            if window is not None:
                EventLoop.poll_and_dispatch()

            raw_evs = evs
            raw_n = int(len(raw_evs))
            current_sync_index_for_mask = ext_trigger_state.get('last_selected_sync_index')
            current_sync_event_ts_for_mask = ext_trigger_state.get('last_selected_event_ts_us')
            # 当前帧人体 mask 前置过滤：只使用同 sync_index 的当前 MVS 人体范围。
            # 但同一 MVS 帧约 40ms，会覆盖多个 4ms event slice，因此同 sync_index 内持久复用当前帧 mask。
            # 过滤后的 detection_evs 才进入 STC / spatter_tracker / line_filter / hit event。
            detection_evs = event_human_mask_filter.filter_events(
                raw_evs,
                current_sync_index=current_sync_index_for_mask,
                current_sync_event_ts_us=current_sync_event_ts_for_mask,
                slice_ts_us=int(ts),
            )
            detection_n = int(len(detection_evs))
            if args.print_event_stats:
                if event_stats_last_ts is None:
                    event_stats_last_ts = int(ts)
                event_stats_acc_raw += raw_n
                dt_us_stats = int(ts) - int(event_stats_last_ts)
                if dt_us_stats >= event_stats_interval_us:
                    raw_mevs = float(event_stats_acc_raw) / float(max(1, dt_us_stats))
                    raw_itemsize = int(getattr(getattr(raw_evs, 'dtype', None), 'itemsize', 0) or 0)
                    raw_mib_s = (event_stats_acc_raw * raw_itemsize) / (dt_us_stats / 1_000_000.0) / (1024.0 * 1024.0) if raw_itemsize > 0 else 0.0
                    now_wall_stats_us = _now_wall_time_us()
                    if event_stats_last_wall_us is None:
                        event_stats_last_wall_us = int(now_wall_stats_us)
                    wall_dt_us_stats = max(1, int(now_wall_stats_us) - int(event_stats_last_wall_us))
                    realtime_ratio = float(dt_us_stats) / float(wall_dt_us_stats)
                    lag_rate_ms = (float(wall_dt_us_stats) - float(dt_us_stats)) / 1000.0
                    print(
                        f"[EventPerf][{camera_log_tag}][SN={camera_sn_for_log}] "
                        f"raw={event_stats_acc_raw} "
                        f"event_dt_ms={dt_us_stats/1000.0:.1f} "
                        f"wall_dt_ms={wall_dt_us_stats/1000.0:.1f} "
                        f"ratio={realtime_ratio:.3f} "
                        f"lag_rate_ms={lag_rate_ms:.1f} "
                        f"raw_rate={raw_mevs:.2f}MEv/s "
                        f"cpp={bool(getattr(args, 'use_cpp_line_filter', False))} "
                        f"overlay={bool(getattr(args, 'enable_event_overlay_pub', False))} "
                        f"mask={bool(getattr(args, 'event_human_mask_filter_enable', False))} "
                        f"mask_wait_ms={float(getattr(args, 'event_human_mask_wait_ms', 0.0) or 0.0):.1f}",
                        flush=True,
                    )
                    event_stats_last_ts = int(ts)
                    event_stats_last_wall_us = int(now_wall_stats_us)
                    event_stats_acc_raw = 0

            dt_us_guard = int(args.accumulation_time_us) if last_loop_ts is None else max(1, int(ts) - int(last_loop_ts))
            last_loop_ts = int(ts)
            shake_active, raw_rate_mevs, shake_entered, shake_exited = shake_guard.update(raw_n, dt_us_guard, ts)
            if shake_entered:
                print(f"[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] ACTIVE raw_rate={raw_rate_mevs:.2f} MEv/s", flush=True)
                # 进入保护时清空检测状态，避免全局抖动的垃圾轨迹污染后续结果
                _reset_realtime_detection_state('shake_guard_enter')
            elif shake_exited:
                print(f"[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] RELEASE raw_rate={raw_rate_mevs:.2f} MEv/s", flush=True)

            if cluster_guard_active and int(ts) >= int(cluster_guard_until_ts):
                cluster_guard_active = False
                cluster_guard_over_count = 0
                print(f"[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] CLUSTER_RELEASE", flush=True)

            # 前置保护：在 spatter_tracker.process_events 之前，直接根据 raw_n / raw_rate 判断是否跳过重检测。
            # 这是真正防止“飞溅物算法先卡死、之后才发现 cluster 爆炸”的关键保护。
            if pre_spatter_guard_enabled:
                pre_over_raw = bool(pre_spatter_raw_thresh > 0 and raw_n >= pre_spatter_raw_thresh)
                pre_over_rate = bool(pre_spatter_rate_thresh > 0 and raw_rate_mevs >= pre_spatter_rate_thresh)
                pre_over = bool(pre_over_raw or pre_over_rate)
                if pre_over:
                    pre_spatter_over_count += 1
                    pre_spatter_last_reason = 'events' if pre_over_raw else 'rate'
                elif not pre_spatter_guard_active:
                    pre_spatter_over_count = 0

                if (not pre_spatter_guard_active) and pre_spatter_over_count >= pre_spatter_trigger_slices:
                    pre_spatter_guard_active = True
                    pre_spatter_guard_until_ts = int(ts) + pre_spatter_hold_us
                    print(
                        f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] PRE_SPATTER_ACTIVE "
                        f"reason={pre_spatter_last_reason} raw_n={raw_n} raw_rate={raw_rate_mevs:.2f} MEv/s "
                        f"max_events={pre_spatter_raw_thresh} rate_thresh={pre_spatter_rate_thresh:.2f}",
                        flush=True,
                    )
                    if bool(getattr(args, 'pre_spatter_reset_on_enter', True)):
                        _reset_realtime_detection_state('pre_spatter_guard_enter')
                elif pre_spatter_guard_active:
                    if pre_over:
                        pre_spatter_guard_until_ts = int(ts) + pre_spatter_hold_us
                    else:
                        release_by_time = int(ts) >= int(pre_spatter_guard_until_ts)
                        release_by_raw = (pre_spatter_raw_thresh <= 0) or (raw_n <= int(pre_spatter_raw_thresh * pre_spatter_release_ratio))
                        release_by_rate = (pre_spatter_rate_thresh <= 0) or (raw_rate_mevs <= pre_spatter_rate_thresh * pre_spatter_release_ratio)
                        if release_by_time and release_by_raw and release_by_rate:
                            pre_spatter_guard_active = False
                            pre_spatter_over_count = 0
                            print(
                                f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] PRE_SPATTER_RELEASE "
                                f"raw_n={raw_n} raw_rate={raw_rate_mevs:.2f} MEv/s",
                                flush=True,
                            )

            raw_record_light_active = bool(getattr(args, 'raw_record_light_mode', False) and raw_recorder.is_recording())
            if raw_record_light_active and not raw_record_light_prev_active:
                print(f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] RAW_RECORD_LIGHT_ACTIVE", flush=True)
                _reset_realtime_detection_state('raw_record_light_enter')
            elif (not raw_record_light_active) and raw_record_light_prev_active:
                print(f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] RAW_RECORD_LIGHT_RELEASE", flush=True)
            raw_record_light_prev_active = raw_record_light_active

            guard_active = bool(shake_active or cluster_guard_active or pre_spatter_guard_active or raw_record_light_active)

            if guard_active:
                # 保护期间只保留最轻量的画面更新；重检测、子弹事件、TSF1 可按开关跳过
                if do_filtering:
                    stc_filter.process_events(raw_evs, filtered_evs)
                buffer = filtered_evs if do_filtering else raw_evs

                if tsf1_publisher is not None and (not args.shake_guard_skip_tsf1) and (not (pre_spatter_guard_active and args.pre_spatter_skip_tsf1)):
                    tsf1_publisher.update_events(buffer)
                    tsf1_publisher.publish_up_to(ts)

                events_frame_gen_algo.process_events(buffer)
                iter_idx += 1
                need_render = (iter_idx % max(1, args.render_every_n) == 0)
                if need_render:
                    raw_event_img = _maybe_render(
                        ts, output_img, events_frame_gen_algo, line_filter,
                        [], input_nozone_rois, window, video_writer,
                        debug_draw_raw_clusters=False,
                        raw_clusters_np=None,
                        raw_recording=raw_recorder.is_recording(),
                        obs_streamer=obs_streamer,
                        submit_obs=False,
                        obs_output_mode=args.obs_output_mode,
                        obs_bullet_only=args.obs_bullet_only,
                        out_video_mode=args.out_video_mode,
                        event_overlay_pub_writer=event_overlay_pub_writer,
                        event_overlay_pub_mode=args.event_overlay_pub_mode,
                    )
                    if raw_record_light_active:
                        guard_label = "RAW_RECORD_LIGHT"
                    elif cluster_guard_active and not shake_active:
                        guard_label = "CLUSTER_GUARD"
                    elif pre_spatter_guard_active and not shake_active:
                        guard_label = "PRE_SPATTER_GUARD"
                    else:
                        guard_label = "SHAKE_GUARD"
                    cv2.putText(output_img, guard_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    if window is not None:
                        window.show_async(output_img)
                    skip_obs_guard = bool((pre_spatter_guard_active and args.pre_spatter_skip_obs) or (raw_record_light_active and args.raw_record_light_skip_obs))
                    if not skip_obs_guard:
                        _submit_obs_frame(
                            obs_streamer, output_img, line_filter, [],
                            obs_output_mode=args.obs_output_mode,
                            raw_event_img=raw_event_img,
                            obs_bullet_only=args.obs_bullet_only,
                            frame_ts_us=ts,
                        )
                if window is not None and window.should_close():
                    break
                continue

            if do_filtering:
                stc_filter.process_events(detection_evs, filtered_evs)
            buffer = filtered_evs if do_filtering else detection_evs

            if tsf1_publisher is not None:
                tsf1_publisher.update_events(buffer)
                tsf1_publisher.publish_up_to(ts)

            events_frame_gen_algo.process_events(buffer)

            if args.disable_spatter_processing:
                # 调试模式：只保留 RAW 录制 / TSF1 / 事件画面，不运行飞溅物与子弹检测。
                iter_idx += 1
                need_render = (iter_idx % max(1, args.render_every_n) == 0)
                if need_render:
                    raw_event_img = _maybe_render(
                        ts, output_img, events_frame_gen_algo, line_filter,
                        [], input_nozone_rois, window, video_writer,
                        debug_draw_raw_clusters=False,
                        raw_clusters_np=None,
                        raw_recording=raw_recorder.is_recording(),
                        obs_streamer=obs_streamer,
                        submit_obs=False,
                        obs_output_mode=args.obs_output_mode,
                        obs_bullet_only=args.obs_bullet_only,
                        out_video_mode=args.out_video_mode,
                        event_overlay_pub_writer=event_overlay_pub_writer,
                        event_overlay_pub_mode=args.event_overlay_pub_mode,
                    )
                    cv2.putText(output_img, "SPATTER_DISABLED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
                    if window is not None:
                        window.show_async(output_img)
                    _submit_obs_frame(
                        obs_streamer, output_img, line_filter, [],
                        obs_output_mode=args.obs_output_mode,
                        raw_event_img=raw_event_img,
                        obs_bullet_only=args.obs_bullet_only,
                        frame_ts_us=ts,
                    )
                if window is not None and window.should_close():
                    break
                continue

            spatter_tracker.process_events(buffer, ts, clusters)

            clusters_np = clusters.numpy()
            cluster_count = int(len(clusters_np))
            tracker_count = cluster_count
            try:
                _tracker_count_attr = getattr(spatter_tracker, 'get_cluster_count', None)
                if callable(_tracker_count_attr):
                    tracker_count = int(_tracker_count_attr())
                elif _tracker_count_attr is not None:
                    tracker_count = int(_tracker_count_attr)
            except Exception:
                tracker_count = cluster_count
            cluster_guard_count = max(cluster_count, tracker_count)

            if cluster_guard_enabled:
                if cluster_guard_count >= cluster_guard_thresh:
                    cluster_guard_over_count += 1
                else:
                    cluster_guard_over_count = 0

                if (not cluster_guard_active) and cluster_guard_over_count >= max(1, int(args.shake_guard_trigger_slices)):
                    cluster_guard_active = True
                    cluster_guard_until_ts = int(ts) + cluster_guard_hold_us
                    print(
                        f"[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] CLUSTER_ACTIVE "
                        f"cluster_count={cluster_guard_count} thresh={cluster_guard_thresh}",
                        flush=True,
                    )
                    # cluster 爆炸说明 spatter 状态已被全局抖动污染，立即清空，避免拖累后续帧。
                    _reset_realtime_detection_state('cluster_guard_enter')

                    iter_idx += 1
                    need_render = (iter_idx % max(1, args.render_every_n) == 0)
                    if need_render:
                        raw_event_img = _maybe_render(
                            ts, output_img, events_frame_gen_algo, line_filter,
                            [], input_nozone_rois, window, video_writer,
                            debug_draw_raw_clusters=False,
                            raw_clusters_np=None,
                            raw_recording=raw_recorder.is_recording(),
                            obs_streamer=obs_streamer,
                            submit_obs=False,
                            obs_output_mode=args.obs_output_mode,
                            obs_bullet_only=args.obs_bullet_only,
                            out_video_mode=args.out_video_mode,
                            event_overlay_pub_writer=event_overlay_pub_writer,
                            event_overlay_pub_mode=args.event_overlay_pub_mode,
                        )
                        cv2.putText(output_img, "CLUSTER_GUARD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                        if window is not None:
                            window.show_async(output_img)
                        _submit_obs_frame(
                            obs_streamer, output_img, line_filter, [],
                            obs_output_mode=args.obs_output_mode,
                            raw_event_img=raw_event_img,
                            obs_bullet_only=args.obs_bullet_only,
                            frame_ts_us=ts,
                        )
                    if window is not None and window.should_close():
                        break
                    continue

            accepted_clusters = line_filter.update(clusters_np, ts)

            for cluster in accepted_clusters:
                log.append([
                    ts,
                    int(cluster['bullet_id']), int(cluster['raw_id']),
                    int(cluster['x']),         int(cluster['y']),
                    int(cluster['width']),     int(cluster['height']),
                ])

            # 提取子弹事件并提交给发送器（主循环里只入队，不做网络 I/O）
            _debug_rows = line_filter.get_last_debug_rows()
            _write_spatter_raw_debug_rows(spatter_raw_debug_writer, ts, clusters_np, accepted_clusters, _debug_rows)
            spatter_summary_stats = _update_spatter_summary(
                spatter_summary_stats, ts, raw_n, cluster_count, len(accepted_clusters),
                int(getattr(args, 'spatter_summary_print_ms', 0) or 0),
                camera_log_tag, camera_sn_for_log,
            )
            try:
                bullet_extractor.set_event_context({
                    'event_human_mask_filter': event_human_mask_filter.get_last_stats(),
                })
            except Exception:
                pass
            bullet_extractor.process(_debug_rows)
            # UE5 渲染实时点流改为使用 Linux 显示层轨迹：
            # line_filter.get_draw_paths(False) 只返回已经进入显示层的 bullet path，
            # 可避免把 debug_rows 中未真正显示的短碎片/候选轨迹推给 UE5。
            try:
                if hasattr(bullet_extractor, "process_display_paths"):
                    _draw_paths = []
                    if hasattr(line_filter, "get_draw_paths"):
                        _draw_paths = line_filter.get_draw_paths(False)
                    else:
                        # Python fallback: at least pass accepted/display clusters as single-point paths.
                        _draw_paths = accepted_clusters
                    bullet_extractor.process_display_paths(_draw_paths)
            except Exception as _e:
                # 不让 UE5 点流调试影响主检测链路。
                if int(getattr(args, 'debug_bullet_event_send', 0) or 0):
                    print(f"[BulletEventExtractor][WARN] process_display_paths failed: {_e}", flush=True)

            if bullet_effect_writer is not None:
                _write_bullet_effect_rows(
                    bullet_effect_writer, _debug_rows,
                    camera_sn=str(args.tsf1_camera_id),
                    camera_alias=camera_short_alias,
                    session_start_camera_ts_us=int(session_start_camera_ts_us or ts),
                    session_start_wall_time_us=int(session_start_wall_time_us),
                    capture_wall_time_us=int(capture_wall_time_us),
                )

            # Skip heavy debugging during throttling
            if debug_csv_writer is not None:
                for row in _debug_rows:
                    debug_csv_writer.writerow(row)

            iter_idx   += 1
            need_render = (iter_idx % max(1, args.render_every_n) == 0)
            
            if need_render:
                _maybe_render(
                    ts, output_img, events_frame_gen_algo, line_filter,
                    accepted_clusters, input_nozone_rois, window, video_writer,
                    debug_draw_raw_clusters=args.debug_draw_raw_clusters,
                    raw_clusters_np=clusters_np,
                    raw_recording=raw_recorder.is_recording(),
                    obs_streamer=obs_streamer,
                    submit_obs=True,
                    obs_output_mode=args.obs_output_mode,
                    obs_bullet_only=args.obs_bullet_only,
                    out_video_mode=args.out_video_mode,
                    event_overlay_pub_writer=event_overlay_pub_writer,
                    event_overlay_pub_mode=args.event_overlay_pub_mode,
                )

            if window is not None and window.should_close():
                break

        try:
            _final_count_attr = getattr(spatter_tracker, 'get_cluster_count', None)
            if callable(_final_count_attr):
                _final_count = _final_count_attr()
            else:
                _final_count = _final_count_attr
        except Exception:
            _final_count = 'unknown'
        print(f'Number of tracked clusters [{camera_log_tag}][SN={camera_sn_for_log}]: {_final_count}')

    finally:
        raw_record_control_service.close()
        raw_recorder.close()
        if video_writer is not None:
            video_writer.release()
            print('Video saved to', video_name)
        if tsf1_sender_proc is not None:
            try:
                tsf1_sender_proc.terminate()
                tsf1_sender_proc.wait(timeout=2.0)
            except Exception:
                try:
                    tsf1_sender_proc.kill()
                except Exception:
                    pass
        if tsf1_pub_writer is not None:
            tsf1_pub_writer.close(unlink=False)
        if event_overlay_pub_writer is not None:
            event_overlay_pub_writer.close(unlink=False)
        if obs_streamer is not None:
            obs_streamer.close()
        bullet_sender.close()
        if debug_csv_file is not None:
            debug_csv_file.close()
        if bullet_effect_file is not None:
            bullet_effect_file.close()
        if spatter_raw_debug_file is not None:
            spatter_raw_debug_file.close()
        if ext_trigger_file is not None:
            ext_trigger_file.close()
        if window is not None:
            window.__exit__(None, None, None)

    if args.out_log:
        out_log_path = _build_alias_output_path(args.out_log, camera_short_alias, args.tsf1_camera_id)
        with open(out_log_path, mode='w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'bullet_id', 'raw_id', 'x', 'y', 'width', 'height'])
            w.writerows(log)


if __name__ == '__main__':
    main()
