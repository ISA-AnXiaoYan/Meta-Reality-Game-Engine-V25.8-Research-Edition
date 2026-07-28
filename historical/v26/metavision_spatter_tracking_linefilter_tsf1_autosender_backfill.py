import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for _p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent / "test607", _THIS_DIR.parent / "test",
           _THIS_DIR.parent / "test-42"]:
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
import copy
import inspect
import json
import re
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
import collections
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_analytics import SpatterTrackerAlgorithm, SpatterTrackingConfig, EventSpatterClusterBuffer
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

try:
    from native_event_bridge.hal_event_pipe_iterator import HalEventFifoIterator, HalEventPipeIterator
except Exception:
    HalEventFifoIterator = None
    HalEventPipeIterator = None

from bullet_event_sender import BulletEventSender, BulletEventExtractor
from line_motion_filter_backfill import (
    LinearMotionFilter,
    TrackConfig,
    AssociationConfig,
    DrawConfig,
)
from event_track_polygon_gate import EventTrackPolygonGate, BusinessIdV1Shadow, BusinessIdV3Shadow

ev_filter_type_dict = {
    "NoFilter": SpatterTrackingConfig.FilterType.NoFilter,
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
        self.state = "NORMAL"  # NORMAL, LIGHT, SEVERE, HOLD

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
            step = int(np.ceil(n / (self.max_eps * 0.5)))  # 更激进的采样
        else:
            self.state = "LIGHT"
            step = int(np.ceil(n / self.max_eps))

        if step > 1:
            return evs[::step]
        return evs


class _SimpleRateLimiter:
    """Wall-clock based max-FPS limiter for optional visualization/stream outputs.

    This is intentionally independent from event timestamps: event timestamps may
    reset or jump on device restart, while output throttling should follow wall
    time.  max_fps<=0 disables limiting.
    """

    def __init__(self, max_fps=0.0, name=''):
        self.name = str(name or '')
        self.max_fps = float(max_fps or 0.0)
        self.min_interval_s = 0.0 if self.max_fps <= 0 else 1.0 / self.max_fps
        self.next_s = 0.0
        self.allowed = 0
        self.skipped = 0

    def due(self):
        if self.min_interval_s <= 0.0:
            return True
        now = time.monotonic()
        return self.next_s <= 0.0 or now >= self.next_s

    def allow(self):
        if self.min_interval_s <= 0.0:
            self.allowed += 1
            return True
        now = time.monotonic()
        if self.next_s <= 0.0:
            self.next_s = now
        if now + 1e-9 >= self.next_s:
            while self.next_s <= now + 1e-9:
                self.next_s += self.min_interval_s
            self.allowed += 1
            return True
        self.skipped += 1
        return False

    def stats(self):
        return {
            'name': self.name,
            'max_fps': float(self.max_fps),
            'allowed': int(self.allowed),
            'skipped': int(self.skipped),
            'next_due_s': float(self.next_s),
        }


class EventLoopStageProfiler:
    """Low-overhead stage profiler for the event worker hot loop.

    It is disabled by default and only writes interval summaries.  The first
    pseudo-stage, ``iterator_gap``, measures time between the previous loop
    finish and the next Python loop body entry, which includes the EventsIterator
    ``next()`` call hidden by the ``for evs in mv_iterator`` syntax.
    """

    def __init__(self, args, camera_alias: str):
        self.enabled = bool(getattr(args, 'event_loop_profiler_enable', False))
        self.camera = str(camera_alias or '')
        self.interval_s = max(0.1, float(getattr(args, 'event_loop_profiler_interval_ms', 1000.0) or 1000.0) / 1000.0)
        tmpl = str(getattr(args, 'event_loop_profiler_jsonl_template', './sync_ipc/event_loop_profile_{camera}.jsonl') or '')
        self.path = ''
        if self.enabled and tmpl:
            try:
                self.path = tmpl.format(camera=self.camera, cam=self.camera, alias=self.camera)
            except Exception:
                self.path = tmpl
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            except Exception:
                pass
        self._last_finish_ns = 0
        self._slice_start_ns = 0
        self._stage_start_ns = 0
        self._period_start_ns = time.perf_counter_ns()
        self._next_report_ns = self._period_start_ns + int(self.interval_s * 1_000_000_000)
        self._stages = collections.OrderedDict()
        self._reasons = collections.Counter()
        self._total_slices = 0
        self._period_slices = 0
        self._fastpath_slices = 0
        self._raw_events = 0
        self._detection_events = 0
        self._last_summary = {'enabled': bool(self.enabled)}

    def start_slice(self):
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        if self._last_finish_ns:
            self._add('iterator_gap', now - self._last_finish_ns)
        self._slice_start_ns = now
        self._stage_start_ns = now

    def mark(self, name: str):
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        self._add(str(name or 'unknown'), now - self._stage_start_ns)
        self._stage_start_ns = now

    def finish(self, reason: str = 'loop', raw_n: int = 0, detection_n: int = 0, fastpath: bool = False):
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        if self._stage_start_ns:
            self._add('loop_tail', now - self._stage_start_ns)
        self._last_finish_ns = now
        self._total_slices += 1
        self._period_slices += 1
        if fastpath:
            self._fastpath_slices += 1
        self._raw_events += int(raw_n or 0)
        self._detection_events += int(detection_n or 0)
        self._reasons[str(reason or 'loop')] += 1
        if now >= self._next_report_ns:
            self.flush(now_ns=now)

    def _add(self, name: str, ns: int):
        if ns < 0:
            ns = 0
        rec = self._stages.get(name)
        if rec is None:
            rec = {'count': 0, 'total_ns': 0, 'max_ns': 0}
            self._stages[name] = rec
        rec['count'] += 1
        rec['total_ns'] += int(ns)
        rec['max_ns'] = max(int(rec['max_ns']), int(ns))

    def _build_summary(self, now_ns: int):
        elapsed_s = max(1e-9, (int(now_ns) - int(self._period_start_ns)) / 1e9)
        stage_total_ns = sum(int(v.get('total_ns', 0)) for v in self._stages.values())
        stages = {}
        for name, rec in self._stages.items():
            count = max(1, int(rec.get('count', 0)))
            total_ns = int(rec.get('total_ns', 0))
            stages[name] = {
                'count': int(rec.get('count', 0)),
                'total_us': round(total_ns / 1000.0, 3),
                'avg_us': round((total_ns / count) / 1000.0, 3),
                'max_us': round(int(rec.get('max_ns', 0)) / 1000.0, 3),
                'share': round(float(total_ns) / float(max(1, stage_total_ns)), 6),
            }
        top = sorted(
            ((name, rec['total_us'], rec['avg_us'], rec['share']) for name, rec in stages.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:12]
        return {
            'enabled': True,
            'camera': self.camera,
            'pid': os.getpid(),
            'wall_time_us': int(time.time() * 1_000_000),
            'elapsed_s': round(elapsed_s, 6),
            'total_slices': int(self._total_slices),
            'period_slices': int(self._period_slices),
            'period_loop_fps': round(float(self._period_slices) / elapsed_s, 3),
            'fastpath_slices': int(self._fastpath_slices),
            'raw_events': int(self._raw_events),
            'detection_events': int(self._detection_events),
            'reasons': dict(self._reasons),
            'stages': stages,
            'top_stages': [
                {'name': name, 'total_us': total_us, 'avg_us': avg_us, 'share': share}
                for name, total_us, avg_us, share in top
            ],
        }

    def flush(self, now_ns=None, force=False):
        if not self.enabled:
            return
        now_ns = int(now_ns or time.perf_counter_ns())
        if (not force) and now_ns < self._next_report_ns:
            return
        summary = self._build_summary(now_ns)
        self._last_summary = summary
        if self.path:
            try:
                with open(self.path, 'a', encoding='utf-8') as fp:
                    fp.write(json.dumps(summary, ensure_ascii=False, separators=(',', ':')) + '\n')
            except Exception as exc:
                self._last_summary = dict(summary)
                self._last_summary['write_error'] = f"{type(exc).__name__}: {exc}"
        self._stages.clear()
        self._reasons.clear()
        self._period_slices = 0
        self._fastpath_slices = 0
        self._raw_events = 0
        self._detection_events = 0
        self._period_start_ns = now_ns
        self._next_report_ns = now_ns + int(self.interval_s * 1_000_000_000)

    def stats(self):
        if not self.enabled:
            return {'enabled': False}
        return dict(self._last_summary)


class EventWorkerStatusWriter:
    """Low-overhead status writer for one event camera process.

    The status file is consumed by the PyOpenGL/CUDA diagnostics window and can
    also be inspected offline.  It never participates in the hot detection path:
    writes are interval-limited and atomic via os.replace().
    """

    def __init__(self, args, camera_alias: str, width: int, height: int):
        self.enabled = bool(getattr(args, 'event_worker_status_enable', True))
        self.camera = str(camera_alias or getattr(args, 'hit_camera_alias', '') or getattr(args, 'tsf1_camera_id', '') or '')
        tmpl = str(getattr(args, 'event_worker_status_file', './sync_ipc/event_worker_{camera}_status.json') or '')
        if not tmpl:
            self.enabled = False
            self.path = ''
        else:
            try:
                self.path = tmpl.format(camera=self.camera, cam=self.camera, alias=self.camera, id=self.camera,
                                        serial=getattr(args, 'tsf1_camera_id', ''))
            except Exception:
                self.path = tmpl
        self.interval_s = max(0.05, float(getattr(args, 'event_worker_status_interval_ms', 1000) or 1000) / 1000.0)
        self.next_s = 0.0
        self.width = int(width)
        self.height = int(height)
        self.total_slices = 0
        self.empty_slices = 0
        self.fastpath_slices = 0
        self.total_events = 0
        self.last_write_s = 0.0
        self.last_slice_s = time.monotonic()
        self.last_total_slices = 0
        self.last_error = ''
        if self.enabled and self.path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            except Exception:
                pass

    def account_slice(self, raw_n=0, fastpath=False):
        self.total_slices += 1
        self.total_events += int(raw_n or 0)
        if int(raw_n or 0) <= 0:
            self.empty_slices += 1
        if fastpath:
            self.fastpath_slices += 1

    def should_update(self, force=False):
        if (not self.enabled) or (not self.path):
            return False
        return bool(force) or time.monotonic() >= self.next_s

    def update(self, *, state='running', active_reason='', ts_us=0, raw_n=0, detection_n=0,
               raw_rate_mevs=0.0, accumulation_time_us=0, fastpath=False, cluster_count=0,
               accepted_count=0, guard_state='OK', human_mask_stats=None, rate_limiters=None,
               extra=None, force=False, account=True):
        if account:
            self.account_slice(raw_n=raw_n, fastpath=fastpath)
        now = time.monotonic()
        if (not self.enabled) or (not self.path):
            return
        if (not force) and now < self.next_s:
            return
        dt = max(1e-6, now - (self.last_write_s or now)) if self.last_write_s else self.interval_s
        slice_delta = max(0, self.total_slices - self.last_total_slices)
        loop_fps = float(slice_delta) / max(1e-6, dt)
        empty_ratio = float(self.empty_slices) / float(max(1, self.total_slices))
        obj = {
            'camera': self.camera,
            'pid': os.getpid(),
            'state': str(state),
            'active_reason': str(active_reason or ''),
            'wall_time_us': int(time.time() * 1_000_000),
            'monotonic_ns': int(time.monotonic_ns()),
            'slice_ts_us': int(ts_us or 0),
            'width': int(self.width),
            'height': int(self.height),
            'loop_fps': float(loop_fps),
            'slice_accumulation_us': int(accumulation_time_us or 0),
            'event_rate_mevs': float(raw_rate_mevs or 0.0),
            'events_per_slice': int(raw_n or 0),
            'detection_events_per_slice': int(detection_n or 0),
            'empty_slice_ratio': float(empty_ratio),
            'empty_slice_count': int(self.empty_slices),
            'total_slice_count': int(self.total_slices),
            'fastpath_slice_count': int(self.fastpath_slices),
            'fastpath': bool(fastpath),
            'cluster_count': int(cluster_count or 0),
            'accepted_count': int(accepted_count or 0),
            'guard_state': str(guard_state or 'OK'),
            'cpu_affinity': sorted(list(os.sched_getaffinity(0))) if hasattr(os, 'sched_getaffinity') else [],
            'human_mask': dict(human_mask_stats or {}),
            'rate_limiters': {k: v.stats() for k, v in (rate_limiters or {}).items() if hasattr(v, 'stats')},
            'last_error': str(self.last_error or ''),
        }
        if isinstance(extra, dict):
            obj.update(extra)
        try:
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
            os.replace(tmp, self.path)
            self.last_error = ''
            self.last_write_s = now
            self.last_total_slices = self.total_slices
            self.next_s = now + self.interval_s
        except Exception as exc:
            self.last_error = repr(exc)



@dataclass
class EventVizPacket:
    """Compact visualization packet for post-human-mask event overlay/OBS/TSF1 output.

    This packet is deliberately separate from the hit/bullet path.  It is only
    used by optional visualization outputs and is pushed to a bounded buffer in
    a non-blocking way.
    """
    camera: str
    seq: int
    source_stage: str
    events: object
    event_ts_min_us: int = 0
    event_ts_max_us: int = 0
    event_ts_mid_us: int = 0
    event_wall_us: int = 0
    event_mono_ns: int = 0
    trigger_seq: int = -1
    program_sync_index: int = -1
    mvs_frame_id: int = -1
    human_frame_id: int = -1
    human_mvs_frame_id: int = -1
    human_capture_wall_us: int = 0
    human_record_wall_us: int = 0
    human_result_age_ms: float = -1.0
    event_to_human_capture_delta_ms: float = float('nan')
    event_to_mvs_delta_ms: float = float('nan')
    raw_event_count: int = 0
    filtered_event_count: int = 0
    track_event_count: int = 0
    viz_event_count: int = 0
    mask_valid: bool = False
    mask_source: str = ''
    mask_person_count: int = 0
    mask_age_ms: float = -1.0
    mask_stale: bool = False
    filter_policy: str = 'post_human_mask'
    track_input_policy: str = 'post_mask'
    filter_reduction_ratio: float = 0.0
    empty_after_filter: bool = False
    empty_after_viz: bool = False
    fastpath_reason: str = ''
    push_wall_us: int = 0
    push_mono_ns: int = 0

    def to_meta(self):
        return {
            'camera': self.camera,
            'seq': int(self.seq),
            'source_stage': self.source_stage,
            'event_ts_min_us': int(self.event_ts_min_us),
            'event_ts_max_us': int(self.event_ts_max_us),
            'event_ts_mid_us': int(self.event_ts_mid_us),
            'event_wall_us': int(self.event_wall_us),
            'event_mono_ns': int(self.event_mono_ns),
            'trigger_seq': int(self.trigger_seq),
            'program_sync_index': int(self.program_sync_index),
            'mvs_frame_id': int(self.mvs_frame_id),
            'human_frame_id': int(self.human_frame_id),
            'human_mvs_frame_id': int(self.human_mvs_frame_id),
            'human_capture_wall_us': int(self.human_capture_wall_us),
            'human_record_wall_us': int(self.human_record_wall_us),
            'human_result_age_ms': float(self.human_result_age_ms),
            'event_to_human_capture_delta_ms': float(self.event_to_human_capture_delta_ms) if np.isfinite(self.event_to_human_capture_delta_ms) else None,
            'event_to_mvs_delta_ms': float(self.event_to_mvs_delta_ms) if np.isfinite(self.event_to_mvs_delta_ms) else None,
            'raw_event_count': int(self.raw_event_count),
            'filtered_event_count': int(self.filtered_event_count),
            'track_event_count': int(self.track_event_count),
            'viz_event_count': int(self.viz_event_count),
            'mask_valid': bool(self.mask_valid),
            'mask_source': str(self.mask_source or ''),
            'mask_person_count': int(self.mask_person_count),
            'mask_age_ms': float(self.mask_age_ms),
            'mask_stale': bool(self.mask_stale),
            'filter_policy': str(self.filter_policy),
            'track_input_policy': str(self.track_input_policy),
            'filter_reduction_ratio': float(self.filter_reduction_ratio),
            'empty_after_filter': bool(self.empty_after_filter),
            'empty_after_viz': bool(self.empty_after_viz),
            'fastpath_reason': str(self.fastpath_reason or ''),
            'push_wall_us': int(self.push_wall_us),
            'push_mono_ns': int(self.push_mono_ns),
        }


class EventVizBuffer:
    """Bounded latest-only visualization buffer.

    The event hot loop calls try_push() and never blocks.  The publisher takes
    only the latest packet and clears older packets, so overlay/OBS never tries
    to catch up by replaying stale visualization frames.
    """
    def __init__(self, buffer_ms=50.0, slots=16, drop_policy='drop_oldest'):
        self.buffer_ms = float(buffer_ms or 50.0)
        self.slots = max(1, int(slots or 16))
        self.drop_policy = str(drop_policy or 'drop_oldest')
        self._q = collections.deque(maxlen=self.slots)
        self._lock = threading.Lock()
        self.push_count = 0
        self.drop_count = 0
        self.pop_count = 0
        self.stale_skip_count = 0
        self.last_push_mono_ns = 0
        self.last_pop_mono_ns = 0

    def try_push(self, packet: EventVizPacket) -> bool:
        if packet is None:
            return False
        with self._lock:
            if len(self._q) >= self.slots:
                self.drop_count += 1
            self._q.append(packet)
            self.push_count += 1
            self.last_push_mono_ns = int(time.monotonic_ns())
        return True

    def get_latest(self):
        with self._lock:
            if not self._q:
                return None
            pkt = self._q[-1]
            self._q.clear()
            self.pop_count += 1
            self.last_pop_mono_ns = int(time.monotonic_ns())
            return pkt

    def mark_stale_skip(self):
        with self._lock:
            self.stale_skip_count += 1

    def stats(self):
        now = int(time.monotonic_ns())
        with self._lock:
            size = len(self._q)
            if self._q:
                latest_age_ms = (now - int(self._q[-1].push_mono_ns or 0)) / 1e6
                oldest_age_ms = (now - int(self._q[0].push_mono_ns or 0)) / 1e6
            else:
                latest_age_ms = -1.0
                oldest_age_ms = -1.0
            return {
                'enable': True,
                'buffer_ms': float(self.buffer_ms),
                'slots': int(self.slots),
                'size': int(size),
                'drop_policy': self.drop_policy,
                'push_count': int(self.push_count),
                'drop_count': int(self.drop_count),
                'pop_count': int(self.pop_count),
                'stale_skip_count': int(self.stale_skip_count),
                'latest_age_ms': float(latest_age_ms),
                'oldest_age_ms': float(oldest_age_ms),
                'last_push_age_ms': float((now - self.last_push_mono_ns) / 1e6) if self.last_push_mono_ns else -1.0,
                'last_pop_age_ms': float((now - self.last_pop_mono_ns) / 1e6) if self.last_pop_mono_ns else -1.0,
            }


class EventVizPublisher(threading.Thread):
    """Async publisher for post-human-mask visualization outputs.

    It owns a separate frame generation algorithm.  It publishes only filtered
    events from EventVizPacket and never participates in bullet UDP/hit judge.
    """
    def __init__(self, *, args, camera_alias, width, height, buffer: EventVizBuffer,
                 event_overlay_pub_writer=None, obs_streamer=None, tsf1_publisher=None,
                 rate_limiters=None):
        super().__init__(daemon=True)
        self.args = args
        self.camera = str(camera_alias or '')
        self.width = int(width)
        self.height = int(height)
        self.buffer = buffer
        self.event_overlay_pub_writer = event_overlay_pub_writer
        self.obs_streamer = obs_streamer
        self.tsf1_publisher = tsf1_publisher
        self.rate_limiters = rate_limiters or {}
        self.max_output_age_ms = max(1.0, float(getattr(args, 'event_viz_max_output_age_ms', 50.0) or 50.0))
        self.idle_heartbeat_fps = max(0.0, float(getattr(args, 'event_viz_idle_heartbeat_fps', 20.0) or 0.0))
        self.overlay_render_mode = str(getattr(args, 'event_overlay_render_mode', 'bgr') or 'bgr').lower()
        self.overlay_mono_color = str(getattr(args, 'event_overlay_mono_color', 'white') or 'white').lower()
        self.overlay_mono_decay_us = max(0, int(getattr(args, 'event_overlay_mono_decay_us', 0) or 0))
        self.overlay_draw_debug_text = bool(getattr(args, 'event_overlay_draw_debug_text', True))
        self._stop_evt = threading.Event()
        self._frame_algo = OnDemandFrameGenerationAlgorithm(self.width, self.height, int(getattr(args, 'accumulation_time_us', 4000) or 4000))
        self._frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._mono_frame = np.zeros((self.height, self.width), dtype=np.uint8)
        self._last_render_seq = None
        self._last_render_frame = None
        self._last_render_mode = ''
        self._last_pkt_seq = None
        self._last_heartbeat_s = 0.0
        self._stats_lock = threading.Lock()
        self._stats = {
            'enable': True,
            'worker_mode': str(getattr(args, 'event_viz_worker_mode', 'thread') or 'thread'),
            'source_stage': 'post_human_mask',
            'publish_count': 0,
            'overlay_publish_count': 0,
            'obs_publish_count': 0,
            'tsf1_publish_count': 0,
            'overlay_skip_count': 0,
            'obs_skip_count': 0,
            'tsf1_skip_count': 0,
            'stale_skip_count': 0,
            'mono_render_count': 0,
            'bgr_render_count': 0,
            'last_publish_wall_us': 0,
            'last_error': '',
            'last_packet': {},
        }
        self.sidecar_template = str(getattr(args, 'event_viz_sidecar_template', './sync_ipc/event_viz_{camera}_latest.json') or '')
        self.overlay_sidecar_template = str(getattr(args, 'event_overlay_sidecar_template', './sync_ipc/event_overlay_{camera}_latest.json') or '')
        self.obs_sidecar_template = str(getattr(args, 'event_obs_sidecar_template', './sync_ipc/event_obs_{camera}_latest.json') or '')
        self.obs_sidecar_enable = bool(getattr(args, 'event_obs_sidecar_enable', True))

    def stop(self):
        self._stop_evt.set()

    def stats(self):
        with self._stats_lock:
            d = dict(self._stats)
        return d

    def _set_stat(self, **kwargs):
        with self._stats_lock:
            self._stats.update(kwargs)

    def _inc_stat(self, key, inc=1):
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0) or 0) + int(inc)

    def _format_path(self, tmpl: str):
        if not tmpl:
            return ''
        try:
            return tmpl.format(camera=self.camera, cam=self.camera, alias=self.camera)
        except Exception:
            return tmpl

    @staticmethod
    def _write_json_atomic(path: str, obj: dict):
        if not path:
            return
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
            os.replace(tmp, path)
        except Exception:
            pass

    def _render_packet(self, pkt: EventVizPacket) -> np.ndarray:
        self._frame[:] = 0
        evs = pkt.events
        if evs is not None and len(evs) > 0:
            try:
                self._frame_algo.process_events(evs)
                self._frame_algo.generate(int(pkt.event_ts_max_us or pkt.event_ts_mid_us or 0), self._frame)
            except Exception as exc:
                self._set_stat(last_error=f'render:{type(exc).__name__}:{exc}')
                self._frame[:] = 0
        if self.overlay_draw_debug_text:
            try:
                raw_n = int(pkt.raw_event_count)
                filt_n = int(pkt.filtered_event_count)
                reduction = float(pkt.filter_reduction_ratio) * 100.0
                mask_s = 'Y' if pkt.mask_valid else 'N'
                txt = f"{self.camera} post-mask raw={raw_n} filt={filt_n} red={reduction:.1f}% mask={mask_s}"
                cv2.putText(self._frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                pass
        self._inc_stat('bgr_render_count')
        return self._frame

    def _render_mono_packet(self, pkt: EventVizPacket) -> np.ndarray:
        self._mono_frame[:] = 0
        evs = pkt.events
        if evs is not None and len(evs) > 0:
            try:
                xs = evs['x'].astype(np.int32, copy=False)
                ys = evs['y'].astype(np.int32, copy=False)
                valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
                if np.any(valid):
                    xv = xs[valid]
                    yv = ys[valid]
                    vals = None
                    dtype_names = getattr(getattr(evs, 'dtype', None), 'names', ()) or ()
                    if self.overlay_mono_decay_us > 0 and 't' in dtype_names:
                        ts_ref = int(pkt.event_ts_max_us or pkt.event_ts_mid_us or 0)
                        tv = evs['t'][valid].astype(np.int64, copy=False)
                        age = np.maximum(0, ts_ref - tv)
                        vals = np.clip(255.0 * (1.0 - age.astype(np.float32) / float(self.overlay_mono_decay_us)), 1, 255).astype(np.uint8)
                    if vals is None:
                        self._mono_frame[yv, xv] = 255
                    else:
                        np.maximum.at(self._mono_frame, (yv, xv), vals)
            except Exception as exc:
                self._set_stat(last_error=f'mono_render:{type(exc).__name__}:{exc}')
                self._mono_frame[:] = 0
        if self.overlay_draw_debug_text:
            try:
                cv2.putText(self._mono_frame, str(self.camera), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)
            except Exception:
                pass
        self._inc_stat('mono_render_count')
        return self._mono_frame

    def _render_for_outputs(self, pkt: EventVizPacket) -> np.ndarray:
        mode = 'mono_sparse' if self.overlay_render_mode in ('mono', 'mono_sparse', 'sparse_mono') else 'bgr'
        if self._last_render_seq == int(pkt.seq) and self._last_render_frame is not None and self._last_render_mode == mode:
            return self._last_render_frame
        frame = self._render_mono_packet(pkt) if mode == 'mono_sparse' else self._render_packet(pkt)
        self._last_render_seq = int(pkt.seq)
        self._last_render_frame = frame
        self._last_render_mode = mode
        return frame

    def _publish_meta(self, pkt: EventVizPacket, kind: str, frame_meta: dict = None):
        meta = pkt.to_meta()
        meta.update({
            'published_wall_us': int(_now_wall_time_us()),
            'published_mono_ns': int(time.monotonic_ns()),
            'schema_version': 2,
            'kind': str(kind),
        })
        if isinstance(frame_meta, dict):
            meta.update(frame_meta)
        if kind == 'overlay':
            self._write_json_atomic(self._format_path(self.overlay_sidecar_template), meta)
            if self.sidecar_template and self.sidecar_template != self.overlay_sidecar_template:
                # Legacy compatibility only.  New preview readers should use
                # event_overlay_{camera}_latest.json so OBS can never overwrite
                # overlay timing diagnostics.
                self._write_json_atomic(self._format_path(self.sidecar_template), meta)
        if kind == 'obs' and self.obs_sidecar_enable:
            self._write_json_atomic(self._format_path(self.obs_sidecar_template), meta)

    def run(self):
        sleep_s = 0.002
        while not self._stop_evt.is_set():
            pkt = self.buffer.get_latest()
            if pkt is None:
                # Optional heartbeat intentionally does not publish empty frames to avoid
                # overwriting the last useful visualization.  It only keeps CPU low.
                time.sleep(sleep_s)
                continue
            now_mono = int(time.monotonic_ns())
            pkt_age_ms = (now_mono - int(pkt.push_mono_ns or now_mono)) / 1e6
            if pkt_age_ms > self.max_output_age_ms:
                self.buffer.mark_stale_skip()
                self._inc_stat('stale_skip_count')
                continue
            overlay_limiter = self.rate_limiters.get('event_overlay')
            obs_limiter = self.rate_limiters.get('obs')
            tsf1_limiter = self.rate_limiters.get('tsf1')
            overlay_due = self.event_overlay_pub_writer is not None and (overlay_limiter is None or overlay_limiter.due())
            obs_due = self.obs_streamer is not None and (obs_limiter is None or obs_limiter.due())
            tsf1_due = self.tsf1_publisher is not None and (tsf1_limiter is None or tsf1_limiter.due())
            if not (overlay_due or obs_due or tsf1_due):
                time.sleep(sleep_s)
                continue
            frame = None
            if overlay_due or obs_due:
                frame = self._render_for_outputs(pkt)
            published_any = False
            try:
                if self.event_overlay_pub_writer is not None:
                    if overlay_due and (overlay_limiter is None or overlay_limiter.allow()):
                        frame_meta = _publish_shared_bgr_frame(self.event_overlay_pub_writer, frame, int(pkt.event_ts_mid_us or pkt.event_ts_max_us or 0))
                        self._inc_stat('overlay_publish_count')
                        self._publish_meta(pkt, 'overlay', frame_meta)
                        published_any = True
                    else:
                        self._inc_stat('overlay_skip_count')
                if self.obs_streamer is not None:
                    if obs_due and (obs_limiter is None or obs_limiter.allow()):
                        _obs_submit_frame(self.obs_streamer, frame.copy(), int(pkt.event_ts_mid_us or pkt.event_ts_max_us or 0))
                        self._inc_stat('obs_publish_count')
                        self._publish_meta(pkt, 'obs')
                        published_any = True
                    else:
                        self._inc_stat('obs_skip_count')
                if self.tsf1_publisher is not None:
                    if tsf1_due and (tsf1_limiter is None or tsf1_limiter.allow()):
                        if pkt.events is not None and len(pkt.events) > 0:
                            self.tsf1_publisher.update_events(pkt.events)
                        self.tsf1_publisher.publish_up_to(int(pkt.event_ts_max_us or pkt.event_ts_mid_us or 0))
                        self._inc_stat('tsf1_publish_count')
                        published_any = True
                    else:
                        self._inc_stat('tsf1_skip_count')
                if published_any:
                    self._inc_stat('publish_count')
                    self._set_stat(last_publish_wall_us=int(_now_wall_time_us()), last_packet=pkt.to_meta(), last_error='')
            except Exception as exc:
                self._set_stat(last_error=f'publish:{type(exc).__name__}:{exc}')
            time.sleep(sleep_s)


def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _event_ts_range(evs, fallback_ts: int = 0):
    try:
        if evs is not None and len(evs) > 0:
            ts_arr = evs['t']
            t_min = int(np.min(ts_arr))
            t_max = int(np.max(ts_arr))
            return t_min, t_max, int((t_min + t_max) // 2)
    except Exception:
        pass
    t = int(fallback_ts or 0)
    return t, t, t


def _event_spatial_diag(evs, width: int, height: int, sample_limit: int = 20000, top_n: int = 5) -> dict:
    """Summarize current-slice event spatial distribution without storing events."""
    diag = {
        'available': False,
        'event_count': 0,
        'sampled': False,
        'sampled_count': 0,
    }
    try:
        n = int(len(evs)) if evs is not None else 0
    except Exception as exc:
        diag['error'] = f'len:{type(exc).__name__}'
        return diag
    diag['event_count'] = int(n)
    if n <= 0:
        diag.update({
            'available': True,
            'unique_pixels': 0,
            'unique_pixel_ratio': 0.0,
            'events_per_unique_pixel': 0.0,
            'top_hot_pixels': [],
            'single_pixel_like': False,
            'hot_pixel_dominant': False,
            'spatial_collapse_suspect': False,
        })
        return diag
    try:
        names = getattr(getattr(evs, 'dtype', None), 'names', None) or ()
        if 'x' not in names or 'y' not in names:
            diag['error'] = 'missing_xy_fields'
            return diag
        step = 1
        if int(sample_limit or 0) > 0 and n > int(sample_limit):
            step = int(np.ceil(float(n) / float(sample_limit)))
        xs = np.asarray(evs['x'][::step], dtype=np.int64)
        ys = np.asarray(evs['y'][::step], dtype=np.int64)
        sampled_count = int(len(xs))
        diag['sampled'] = bool(step > 1)
        diag['sampled_count'] = int(sampled_count)
        if sampled_count <= 0:
            return diag
        w = int(width or 0)
        h = int(height or 0)
        valid = np.ones(sampled_count, dtype=bool)
        if w > 0:
            valid &= (xs >= 0) & (xs < w)
        if h > 0:
            valid &= (ys >= 0) & (ys < h)
        invalid_count = int(sampled_count - int(np.count_nonzero(valid)))
        if invalid_count > 0:
            xs = xs[valid]
            ys = ys[valid]
        valid_count = int(len(xs))
        if valid_count <= 0:
            diag.update({
                'available': True,
                'invalid_event_count': invalid_count,
                'unique_pixels': 0,
                'top_hot_pixels': [],
                'spatial_collapse_suspect': True,
            })
            return diag
        x_min = int(np.min(xs)); x_max = int(np.max(xs))
        y_min = int(np.min(ys)); y_max = int(np.max(ys))
        bbox_w = int(x_max - x_min + 1)
        bbox_h = int(y_max - y_min + 1)
        code_width = max(1, w)
        codes = ys * int(code_width) + xs
        unique_codes, counts = np.unique(codes, return_counts=True)
        unique_pixels = int(len(unique_codes))
        max_count = int(np.max(counts)) if len(counts) else 0
        max_fraction = float(max_count) / float(max(1, valid_count))
        order = np.argsort(counts)[::-1][:max(0, int(top_n or 0))]
        top_hot_pixels = []
        for idx in order:
            code = int(unique_codes[int(idx)])
            cnt = int(counts[int(idx)])
            top_hot_pixels.append({
                'x': int(code % code_width),
                'y': int(code // code_width),
                'count': cnt,
                'fraction': float(cnt) / float(max(1, valid_count)),
            })
        unique_ratio = float(unique_pixels) / float(max(1, valid_count))
        events_per_unique = float(valid_count) / float(max(1, unique_pixels))
        single_pixel_like = bool(unique_pixels <= 2 and valid_count >= 8)
        hot_pixel_dominant = bool(max_fraction >= 0.50 and valid_count >= 16)
        spatial_collapse = bool(
            (unique_pixels <= 3 and valid_count >= 16)
            or (unique_ratio <= 0.02 and valid_count >= 100)
            or hot_pixel_dominant
        )
        diag.update({
            'available': True,
            'sample_step': int(step),
            'invalid_event_count': int(invalid_count),
            'valid_event_count': int(valid_count),
            'unique_pixels': int(unique_pixels),
            'unique_pixel_ratio': float(unique_ratio),
            'events_per_unique_pixel': float(events_per_unique),
            'bbox': {
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max),
                'width': int(bbox_w),
                'height': int(bbox_h),
                'area': int(max(0, bbox_w) * max(0, bbox_h)),
            },
            'max_pixel_count': int(max_count),
            'max_pixel_fraction': float(max_fraction),
            'top_hot_pixels': top_hot_pixels,
            'single_pixel_like': bool(single_pixel_like),
            'hot_pixel_dominant': bool(hot_pixel_dominant),
            'spatial_collapse_suspect': bool(spatial_collapse),
        })
    except Exception as exc:
        diag['error'] = f'{type(exc).__name__}: {exc}'
    return diag


def _cluster_spatial_diag(clusters_np, accepted_clusters, width: int, height: int, top_n: int = 5) -> dict:
    diag = {
        'available': False,
        'cluster_count': 0,
        'accepted_count': 0,
        'accepted_ratio': 0.0,
        'clusters': [],
    }
    try:
        cluster_count = int(len(clusters_np)) if clusters_np is not None else 0
    except Exception as exc:
        diag['error'] = f'len:{type(exc).__name__}'
        return diag
    try:
        accepted_count = int(len(accepted_clusters or []))
    except Exception:
        accepted_count = 0
    diag['cluster_count'] = int(cluster_count)
    diag['accepted_count'] = int(accepted_count)
    diag['accepted_ratio'] = float(accepted_count) / float(max(1, cluster_count))
    if cluster_count <= 0:
        diag['available'] = True
        return diag
    accepted_ids = set()
    for c in (accepted_clusters or []):
        try:
            accepted_ids.add(int(c['raw_id']))
        except Exception:
            pass
    try:
        rows = []
        invalid_rows = []
        x0s = []; y0s = []; x1s = []; y1s = []
        for c in clusters_np:
            raw_id = _safe_int(c['id'], -1)
            x = _safe_float(c['x'], 0.0)
            y = _safe_float(c['y'], 0.0)
            cw = _safe_float(c['width'], 0.0)
            ch = _safe_float(c['height'], 0.0)
            area = float(max(0.0, cw) * max(0.0, ch))
            valid_cluster = bool(
                np.isfinite(x) and np.isfinite(y) and np.isfinite(cw) and np.isfinite(ch)
                and cw > 0.0 and ch > 0.0
                and x > -float(max(1, width)) and y > -float(max(1, height))
                and x < float(max(1, width) * 2) and y < float(max(1, height) * 2)
            )
            row = {
                'raw_id': int(raw_id),
                'x': float(x),
                'y': float(y),
                'width': float(cw),
                'height': float(ch),
                'cx': float(x + cw * 0.5),
                'cy': float(y + ch * 0.5),
                'area': float(area),
                'accepted': bool(raw_id in accepted_ids),
                'valid': bool(valid_cluster),
            }
            if not valid_cluster:
                invalid_rows.append(row)
                continue
            x0s.append(float(x)); y0s.append(float(y)); x1s.append(float(x + cw)); y1s.append(float(y + ch))
            rows.append(row)
        rows = sorted(rows, key=lambda r: float(r.get('area', 0.0)), reverse=True)
        diag['clusters'] = rows[:max(0, int(top_n or 0))]
        diag['valid_cluster_count'] = int(len(rows))
        diag['invalid_cluster_count'] = int(len(invalid_rows))
        diag['invalid_cluster_samples'] = invalid_rows[:3]
        if x0s:
            diag['cluster_bbox'] = {
                'x_min': float(min(x0s)),
                'y_min': float(min(y0s)),
                'x_max': float(max(x1s)),
                'y_max': float(max(y1s)),
                'width': float(max(x1s) - min(x0s)),
                'height': float(max(y1s) - min(y0s)),
            }
        diag['all_clusters_rejected'] = bool(cluster_count > 0 and accepted_count <= 0)
        diag['available'] = True
    except Exception as exc:
        diag['error'] = f'{type(exc).__name__}: {exc}'
    return diag


def _line_filter_reject_diag(debug_rows) -> dict:
    diag = {
        'available': False,
        'row_count': 0,
        'reject_reason_counts': {},
        'phase_counts': {},
        'assign_kind_counts': {},
    }
    if debug_rows is None:
        diag['error'] = 'debug_rows_unavailable'
        return diag
    try:
        rows = list(debug_rows or [])
        diag['row_count'] = int(len(rows))
        reject_counts = {}
        phase_counts = {}
        assign_counts = {}
        active_rows = 0
        display_rows = 0
        accepted_rows = 0
        max_total_disp = 0.0
        max_recent_disp = 0.0
        max_valid_steps = 0
        sample_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            phase = str(row.get('phase', '') or '')
            assign = str(row.get('assign_kind', '') or '')
            reason = str(row.get('reject_reason', '') or '')
            if not reason:
                reason = 'accepted' if _safe_int(row.get('accepted_now', 0), 0) else 'no_reject_reason'
            reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1
            if phase:
                phase_counts[phase] = int(phase_counts.get(phase, 0)) + 1
            if assign:
                assign_counts[assign] = int(assign_counts.get(assign, 0)) + 1
            if bool(_safe_int(row.get('bullet_active', row.get('accepted_now', 0)), 0)):
                active_rows += 1
            if _safe_int(row.get('display_bullet_id', -1), -1) >= 0:
                display_rows += 1
            if _safe_int(row.get('accepted_now', 0), 0):
                accepted_rows += 1
            total_disp = _safe_float(row.get('total_disp_px', 0.0), 0.0)
            recent_disp = _safe_float(row.get('recent_total_disp_px', 0.0), 0.0)
            valid_steps = _safe_int(row.get('valid_step_count', 0), 0)
            max_total_disp = max(max_total_disp, float(total_disp))
            max_recent_disp = max(max_recent_disp, float(recent_disp))
            max_valid_steps = max(max_valid_steps, int(valid_steps))
            if len(sample_rows) < 5:
                sample_rows.append({
                    'raw_id': _safe_int(row.get('raw_id', -1), -1),
                    'bullet_id': _safe_int(row.get('bullet_id', -1), -1),
                    'display_bullet_id': _safe_int(row.get('display_bullet_id', -1), -1),
                    'phase': phase,
                    'assign_kind': assign,
                    'reject_reason': reason,
                    'total_disp_px': float(total_disp),
                    'recent_total_disp_px': float(recent_disp),
                    'valid_step_count': int(valid_steps),
                })
        diag.update({
            'available': True,
            'reject_reason_counts': dict(sorted(reject_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]),
            'phase_counts': dict(sorted(phase_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]),
            'assign_kind_counts': dict(sorted(assign_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]),
            'active_rows': int(active_rows),
            'display_rows': int(display_rows),
            'accepted_rows': int(accepted_rows),
            'max_total_disp_px': float(max_total_disp),
            'max_recent_total_disp_px': float(max_recent_disp),
            'max_valid_step_count': int(max_valid_steps),
            'sample_rows': sample_rows,
        })
    except Exception as exc:
        diag['error'] = f'{type(exc).__name__}: {exc}'
    return diag


def _parse_hot_pixel_static_map(spec: str, camera_alias: str) -> List[tuple]:
    cam = str(camera_alias or '').strip().upper()
    raw = str(spec or '').strip()
    if not raw:
        return []
    out: List[tuple] = []

    def _add_pair(x, y):
        try:
            out.append((int(round(float(x))), int(round(float(y)))))
        except Exception:
            pass

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            vals = obj.get(cam, obj.get(cam.lower(), obj.get('*', obj.get('all', []))))
            if isinstance(vals, dict):
                vals = vals.get('pixels', vals.get('points', []))
            if isinstance(vals, (list, tuple)):
                for item in vals:
                    if isinstance(item, dict):
                        _add_pair(item.get('x'), item.get('y'))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        _add_pair(item[0], item[1])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, dict):
                    if str(item.get('camera', item.get('cam', cam))).strip().upper() in (cam, '*', 'ALL'):
                        _add_pair(item.get('x'), item.get('y'))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    _add_pair(item[0], item[1])
    except Exception:
        for part in re.split(r'[;|]+', raw):
            token = part.strip()
            if not token:
                continue
            target = ''
            coords = token
            if ':' in token:
                target, coords = token.split(':', 1)
                target = target.strip().upper()
            if target and target not in (cam, '*', 'ALL'):
                continue
            nums = [x for x in re.split(r'[,\s/]+', coords.strip()) if x]
            for i in range(0, len(nums) - 1, 2):
                _add_pair(nums[i], nums[i + 1])
    seen = set()
    dedup = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


class EventHotPixelFilter:
    """Tracking-only hot-pixel suppressor for event worker diagnostics/fixes."""

    def __init__(self, args, camera_alias: str):
        self.camera = str(camera_alias or '').strip().upper()
        self.enabled = bool(getattr(args, 'event_hot_pixel_filter_enable', False))
        self.mode = str(getattr(args, 'event_hot_pixel_filter_mode', 'tracking_only') or 'tracking_only').strip().lower()
        self.radius_px = max(0, int(getattr(args, 'event_hot_pixel_radius_px', 0) or 0))
        self.learn_enable = bool(getattr(args, 'event_hot_pixel_learn_enable', False))
        self.learn_min_fraction = max(0.0, float(getattr(args, 'event_hot_pixel_learn_min_fraction', 0.25) or 0.25))
        self.learn_min_count = max(1, int(getattr(args, 'event_hot_pixel_learn_min_count', 64) or 64))
        self.learn_confirm_slices = max(1, int(getattr(args, 'event_hot_pixel_learn_confirm_slices', 5) or 5))
        self.max_learned = max(0, int(getattr(args, 'event_hot_pixel_learn_max_pixels', 16) or 16))
        self.static_pixels = set(_parse_hot_pixel_static_map(
            str(getattr(args, 'event_hot_pixel_static_map', '') or ''),
            self.camera,
        ))
        self.learned_pixels = set()
        self._streaks: Dict[tuple, int] = {}
        self._learned_updates = 0
        self._last_stats = {
            'enabled': bool(self.enabled),
            'mode': self.mode,
            'radius_px': int(self.radius_px),
            'static_hot_pixels': [{'x': int(x), 'y': int(y)} for x, y in sorted(self.static_pixels)],
            'learned_hot_pixels': [],
            'active_hot_pixels': [{'x': int(x), 'y': int(y)} for x, y in sorted(self.static_pixels)],
            'input_count': 0,
            'output_count': 0,
            'removed_count': 0,
            'removed_ratio': 0.0,
        }

    def _active_pixels(self):
        return set(self.static_pixels) | set(self.learned_pixels)

    def _learn_from_diag(self, spatial_diag: dict) -> Optional[tuple]:
        if (not self.enabled) or (not self.learn_enable):
            return None
        top = spatial_diag.get('top_hot_pixels') if isinstance(spatial_diag, dict) else None
        if not top:
            self._streaks.clear()
            return None
        cand = top[0] if isinstance(top[0], dict) else {}
        try:
            pixel = (int(cand.get('x')), int(cand.get('y')))
            count = int(cand.get('count') or 0)
            fraction = float(cand.get('fraction') or 0.0)
        except Exception:
            return None
        if count < self.learn_min_count or fraction < self.learn_min_fraction:
            self._streaks[pixel] = 0
            return None
        if pixel in self._active_pixels():
            return pixel
        streak = int(self._streaks.get(pixel, 0)) + 1
        self._streaks[pixel] = streak
        if streak >= self.learn_confirm_slices and len(self.learned_pixels) < self.max_learned:
            self.learned_pixels.add(pixel)
            self._learned_updates += 1
            return pixel
        return None

    def filter(self, evs, spatial_diag: Optional[dict] = None):
        stats = {
            'enabled': bool(self.enabled),
            'mode': self.mode,
            'radius_px': int(self.radius_px),
            'learn_enable': bool(self.learn_enable),
            'learn_min_fraction': float(self.learn_min_fraction),
            'learn_min_count': int(self.learn_min_count),
            'learn_confirm_slices': int(self.learn_confirm_slices),
            'learned_updates': int(self._learned_updates),
            'static_hot_pixels': [{'x': int(x), 'y': int(y)} for x, y in sorted(self.static_pixels)],
            'learned_hot_pixels': [{'x': int(x), 'y': int(y)} for x, y in sorted(self.learned_pixels)],
            'active_hot_pixels': [{'x': int(x), 'y': int(y)} for x, y in sorted(self._active_pixels())],
            'input_count': 0,
            'output_count': 0,
            'removed_count': 0,
            'removed_ratio': 0.0,
            'applied_to': 'tracking_only',
        }
        try:
            n = int(len(evs)) if evs is not None else 0
        except Exception:
            n = 0
        stats['input_count'] = int(n)
        if spatial_diag is None and self.learn_enable:
            spatial_diag = _event_spatial_diag(evs, 0, 0)
        elif spatial_diag is None:
            spatial_diag = {}
        learned_pixel = self._learn_from_diag(spatial_diag)
        if learned_pixel is not None:
            stats['latest_learn_candidate'] = {'x': int(learned_pixel[0]), 'y': int(learned_pixel[1])}
        active = self._active_pixels()
        stats['learned_updates'] = int(self._learned_updates)
        stats['learned_hot_pixels'] = [{'x': int(x), 'y': int(y)} for x, y in sorted(self.learned_pixels)]
        stats['active_hot_pixels'] = [{'x': int(x), 'y': int(y)} for x, y in sorted(active)]
        if (not self.enabled) or self.mode not in ('tracking_only', 'track', 'tracking') or n <= 0 or not active:
            stats['output_count'] = int(n)
            self._last_stats = stats
            return evs, stats
        try:
            xs = np.asarray(evs['x'], dtype=np.int64)
            ys = np.asarray(evs['y'], dtype=np.int64)
            keep = np.ones(int(n), dtype=bool)
            r = int(self.radius_px)
            for x, y in active:
                if r <= 0:
                    keep &= ~((xs == int(x)) & (ys == int(y)))
                else:
                    keep &= ~((np.abs(xs - int(x)) <= r) & (np.abs(ys - int(y)) <= r))
            out = evs[keep]
            removed = int(n - int(len(out)))
            stats['output_count'] = int(len(out))
            stats['removed_count'] = int(removed)
            stats['removed_ratio'] = float(removed) / float(max(1, n))
            if removed > 0:
                stats['last_removed_hot_pixels'] = stats['active_hot_pixels']
            self._last_stats = stats
            return out, stats
        except Exception as exc:
            stats['error'] = f'{type(exc).__name__}: {exc}'
            stats['output_count'] = int(n)
            self._last_stats = stats
            return evs, stats

    def stats(self):
        return dict(self._last_stats)


def _sanitize_clusters_for_line_filter(clusters_np, width: int, height: int, enabled: bool = True) -> tuple:
    try:
        count = int(len(clusters_np)) if clusters_np is not None else 0
    except Exception:
        count = 0
    stats = {
        'enabled': bool(enabled),
        'input_count': int(count),
        'output_count': int(count),
        'invalid_cluster_count': 0,
        'removed_count': 0,
        'invalid_cluster_samples': [],
    }
    if (not enabled) or clusters_np is None or count <= 0:
        return clusters_np, stats
    try:
        xs = np.asarray(clusters_np['x'], dtype=np.float64)
        ys = np.asarray(clusters_np['y'], dtype=np.float64)
        ws = np.asarray(clusters_np['width'], dtype=np.float64)
        hs = np.asarray(clusters_np['height'], dtype=np.float64)
        w = max(1.0, float(width or 1))
        h = max(1.0, float(height or 1))
        valid = (
            np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ws) & np.isfinite(hs)
            & (ws > 0.0) & (hs > 0.0)
            & (xs > -w) & (ys > -h)
            & (xs < w * 2.0) & (ys < h * 2.0)
        )
        invalid_idx = np.where(~valid)[0]
        stats['invalid_cluster_count'] = int(len(invalid_idx))
        stats['removed_count'] = int(len(invalid_idx))
        stats['output_count'] = int(count - len(invalid_idx))
        samples = []
        for idx in invalid_idx[:3]:
            try:
                c = clusters_np[int(idx)]
                samples.append({
                    'raw_id': _safe_int(c['id'], -1),
                    'x': _safe_float(c['x'], 0.0),
                    'y': _safe_float(c['y'], 0.0),
                    'width': _safe_float(c['width'], 0.0),
                    'height': _safe_float(c['height'], 0.0),
                })
            except Exception:
                pass
        stats['invalid_cluster_samples'] = samples
        if len(invalid_idx) <= 0:
            return clusters_np, stats
        return clusters_np[valid], stats
    except Exception as exc:
        stats['error'] = f'{type(exc).__name__}: {exc}'
        return clusters_np, stats


def _copy_events_for_viz(evs):
    try:
        if evs is None:
            return evs
        return evs.copy()
    except Exception:
        return evs


def _apply_event_worker_affinity(args, camera_alias: str) -> None:
    spec = str(getattr(args, 'event_worker_cpus', '') or '').strip()
    if not spec:
        return
    try:
        cpus = [int(x.strip()) for x in spec.split(',') if x.strip()]
        if not cpus:
            return
        alias = str(camera_alias or '').strip().upper()
        idx = ord(alias[0]) - ord('A') if alias and alias[0].isalpha() else 0
        cpu = cpus[idx % len(cpus)]
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, {int(cpu)})
            print(f"[EventCPU][{alias}] affinity -> CPU {cpu}", flush=True)
    except Exception as exc:
        print(f"[EventCPU][{camera_alias}][WARN] set affinity failed: {exc}", flush=True)


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
        return self.enabled and self.device is not None and getattr(self.device, 'get_i_events_stream',
                                                                    None) is not None and self.device.get_i_events_stream() is not None

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
                print('[RAW] async control service did not stop within 2s; recorder will reject new starts.',
                      flush=True)


_EVENT_SLICE_MAGIC = b'EVP1'
_EVENT_SLICE_VERSION = 1
_EVENT_SLICE_HEADER = struct.Struct('<4sHHQQQII')


def _raw_record_safe_name(value: str, fallback: str = 'unknown') -> str:
    value = str(value or '').strip() or str(fallback)
    value = re.sub(r'[^0-9A-Za-z_.-]+', '_', value)
    return value[:96] or str(fallback)


def _dtype_descr_json(dtype: np.dtype) -> List[Any]:
    return [[str(name), str(fmt)] for name, fmt in np.dtype(dtype).descr]


class EventSlicePackRecorder:
    """Asynchronously records the current event-slice stream for replay checks.

    This records the post-HAL-FIFO event frames consumed by the worker, not the
    old Metavision ``log_raw_data`` stream.  It keeps the V6 native FIFO path in
    place and only adds a bounded non-blocking queue in the Python hot loop.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        output_root: str = './recordings',
        camera_id: str = 'cam0',
        serial: str = '',
        width: int = 0,
        height: int = 0,
        queue_limit: int = 512,
        flush_interval_ms: int = 1000,
    ):
        self.enabled = bool(enabled)
        self.output_root = str(output_root or './recordings')
        self.camera_id = str(camera_id or 'cam0')
        self.serial = str(serial or '')
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.queue_limit = max(8, int(queue_limit or 512))
        self.flush_interval_s = max(0.1, float(flush_interval_ms or 1000) / 1000.0)
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_limit)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._recording = False
        self._closed = False
        self._session_id = ''
        self._session_dir = ''
        self._data_path = ''
        self._index_path = ''
        self._manifest_path = ''
        self._dtype: Optional[np.dtype] = None
        self._dtype_descr: List[Any] = []
        self._seq = 0
        self._written_slices = 0
        self._written_events = 0
        self._written_bytes = 0
        self._dropped_slices = 0
        self._last_slice_ts_us = 0
        self._last_write_wall_us = 0
        self._last_error = ''

    def can_record(self):
        return bool(self.enabled)

    def is_recording(self):
        with self._lock:
            return bool(self._recording)

    def _build_paths(self, session_id: str):
        session = _raw_record_safe_name(session_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), 'session')
        cam = _raw_record_safe_name(self.camera_id, 'cam')
        serial = _raw_record_safe_name(self.serial, 'serial')
        folder = Path(self.output_root).expanduser() / session / f"EVENT_{cam}_{serial}"
        return (
            session,
            str(folder),
            str(folder / 'event_slices.evp'),
            str(folder / 'event_slices_index.csv'),
            str(folder / 'event_slices_manifest.json'),
        )

    def start(self, session_id=None):
        with self._lock:
            if self._closed:
                self._last_error = 'recorder_closed'
                print(f'[RAW][EVP] start ignored: recorder closed camera={self.camera_id}', flush=True)
                return False
            if not self.enabled:
                self._last_error = 'recorder_disabled'
                print(f'[RAW][EVP] start ignored: recorder disabled camera={self.camera_id}', flush=True)
                return False
            if self._recording:
                return True
            session, session_dir, data_path, index_path, manifest_path = self._build_paths(str(session_id or ''))
            self._session_id = session
            self._session_dir = session_dir
            self._data_path = data_path
            self._index_path = index_path
            self._manifest_path = manifest_path
            self._dtype = None
            self._dtype_descr = []
            self._seq = 0
            self._written_slices = 0
            self._written_events = 0
            self._written_bytes = 0
            self._dropped_slices = 0
            self._last_slice_ts_us = 0
            self._last_write_wall_us = 0
            self._last_error = ''
            self._stop_evt.clear()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            Path(session_dir).mkdir(parents=True, exist_ok=True)
            self._recording = True
            self._thread = threading.Thread(target=self._writer, daemon=True, name=f'EventSliceRecord-{self.camera_id}')
            self._thread.start()
            print(f'[RAW][EVP] recording started camera={self.camera_id} -> {data_path}', flush=True)
            return True

    def stop(self):
        with self._lock:
            if not self._recording:
                return False
            self._recording = False
            self._stop_evt.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=4.0)
            if thread.is_alive():
                self._last_error = 'writer_stop_timeout'
                print(f'[RAW][EVP] writer stop timeout camera={self.camera_id}', flush=True)
        print(f'[RAW][EVP] recording stopped camera={self.camera_id} -> {self._data_path}', flush=True)
        return True

    def toggle(self, session_id=None):
        return self.stop() if self.is_recording() else self.start(session_id=session_id)

    def _write_manifest(self, final: bool = False):
        obj = {
            'schema_version': 1,
            'kind': 'event_slice_pack',
            'camera': self.camera_id,
            'serial': self.serial,
            'session_id': self._session_id,
            'session_dir': self._session_dir,
            'data_file': self._data_path,
            'index_file': self._index_path,
            'width': int(self.width),
            'height': int(self.height),
            'event_dtype_descr': self._dtype_descr,
            'header_format': '<4sHHQQQII',
            'header_size': int(_EVENT_SLICE_HEADER.size),
            'record_magic': _EVENT_SLICE_MAGIC.decode('ascii'),
            'record_version': int(_EVENT_SLICE_VERSION),
            'slices_written': int(self._written_slices),
            'events_written': int(self._written_events),
            'bytes_written': int(self._written_bytes),
            'dropped_slices': int(self._dropped_slices),
            'last_slice_ts_us': int(self._last_slice_ts_us),
            'last_write_wall_us': int(self._last_write_wall_us),
            'final': bool(final),
            'last_error': str(self._last_error or ''),
        }
        try:
            tmp = self._manifest_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._manifest_path)
        except Exception as exc:
            self._last_error = f'manifest_write_failed:{type(exc).__name__}:{exc}'

    def _writer(self):
        data_f = None
        index_f = None
        index_writer = None
        last_flush = time.monotonic()
        try:
            data_f = open(self._data_path, 'ab', buffering=1024 * 1024)
            index_f = open(self._index_path, 'w', newline='', encoding='utf-8')
            index_writer = csv.DictWriter(index_f, fieldnames=[
                'seq', 'slice_ts_us', 'wall_us', 'event_count', 'file_offset',
                'payload_bytes', 'event_local_sync_index', 'mapped_mvs_sync_index',
                'sync_event_ts_us',
            ])
            index_writer.writeheader()
            self._write_manifest(final=False)
            while True:
                item = self._queue.get()
                if item is None:
                    if self._stop_evt.is_set():
                        break
                    continue
                evs, ts_us, wall_us, sync_meta = item
                arr = np.asarray(evs)
                dtype = np.dtype(arr.dtype)
                if self._dtype is None:
                    self._dtype = dtype
                    self._dtype_descr = _dtype_descr_json(dtype)
                    self._write_manifest(final=False)
                elif dtype != self._dtype:
                    with self._lock:
                        self._dropped_slices += 1
                        self._last_error = f'dtype_changed:{dtype}->{self._dtype}'
                    continue
                if not arr.flags['C_CONTIGUOUS']:
                    arr = np.ascontiguousarray(arr)
                payload = arr.tobytes(order='C')
                event_count = int(len(arr))
                seq = int(self._seq)
                offset = int(data_f.tell())
                header = _EVENT_SLICE_HEADER.pack(
                    _EVENT_SLICE_MAGIC,
                    _EVENT_SLICE_VERSION,
                    _EVENT_SLICE_HEADER.size,
                    seq,
                    int(ts_us or 0),
                    int(wall_us or 0),
                    event_count,
                    int(dtype.itemsize),
                )
                data_f.write(header)
                if payload:
                    data_f.write(payload)
                index_writer.writerow({
                    'seq': seq,
                    'slice_ts_us': int(ts_us or 0),
                    'wall_us': int(wall_us or 0),
                    'event_count': event_count,
                    'file_offset': offset,
                    'payload_bytes': len(payload),
                    'event_local_sync_index': sync_meta.get('event_local_sync_index') if isinstance(sync_meta, dict) else '',
                    'mapped_mvs_sync_index': sync_meta.get('mapped_mvs_sync_index') if isinstance(sync_meta, dict) else '',
                    'sync_event_ts_us': sync_meta.get('sync_event_ts_us') if isinstance(sync_meta, dict) else '',
                })
                now = time.monotonic()
                with self._lock:
                    self._seq += 1
                    self._written_slices += 1
                    self._written_events += event_count
                    self._written_bytes += len(header) + len(payload)
                    self._last_slice_ts_us = int(ts_us or 0)
                    self._last_write_wall_us = int(wall_us or 0)
                if now - last_flush >= self.flush_interval_s:
                    data_f.flush()
                    index_f.flush()
                    self._write_manifest(final=False)
                    last_flush = now
        except Exception as exc:
            with self._lock:
                self._last_error = f'{type(exc).__name__}: {exc}'
            print(f'[RAW][EVP] writer error camera={self.camera_id}: {self._last_error}', flush=True)
        finally:
            try:
                if data_f is not None:
                    data_f.flush()
                    data_f.close()
            except Exception:
                pass
            try:
                if index_f is not None:
                    index_f.flush()
                    index_f.close()
            except Exception:
                pass
            self._write_manifest(final=True)

    def submit(self, evs, ts_us: int, wall_us: int, sync_meta: Optional[Dict[str, Any]] = None):
        with self._lock:
            recording = bool(self._recording)
        if not recording:
            return False
        try:
            self._queue.put_nowait((evs, int(ts_us or 0), int(wall_us or 0), dict(sync_meta or {})))
            return True
        except queue.Full:
            with self._lock:
                self._dropped_slices += 1
                self._last_error = 'queue_full'
            return False
        except Exception as exc:
            with self._lock:
                self._dropped_slices += 1
                self._last_error = f'submit_failed:{type(exc).__name__}:{exc}'
            return False

    def status(self):
        with self._lock:
            qsize = 0
            try:
                qsize = int(self._queue.qsize())
            except Exception:
                qsize = 0
            return {
                'enabled': bool(self.enabled),
                'backend': 'event_slice_pack',
                'recording': bool(self._recording),
                'session_id': self._session_id,
                'session_dir': self._session_dir,
                'data_path': self._data_path,
                'index_path': self._index_path,
                'manifest_path': self._manifest_path,
                'queue_size': qsize,
                'queue_limit': int(self.queue_limit),
                'writer_alive': bool(self._thread is not None and self._thread.is_alive()),
                'slices_written': int(self._written_slices),
                'events_written': int(self._written_events),
                'bytes_written': int(self._written_bytes),
                'dropped_slices': int(self._dropped_slices),
                'last_slice_ts_us': int(self._last_slice_ts_us),
                'last_write_wall_us': int(self._last_write_wall_us),
                'last_error': str(self._last_error or ''),
                'event_dtype_descr': list(self._dtype_descr),
                'width': int(self.width),
                'height': int(self.height),
            }

    def close(self):
        with self._lock:
            self._closed = True
        self.stop()


class CombinedEventRawRecorder:
    """Control old HAL RAW logging and the V0.2 replay pack as one recorder."""

    def __init__(self, recorders: List[Any], camera_id: str):
        self.recorders = [r for r in recorders if r is not None]
        self.camera_id = str(camera_id or 'cam0')
        self.enabled = any(bool(getattr(r, 'enabled', False)) for r in self.recorders)

    def can_record(self):
        return bool(self.enabled)

    def start(self, session_id=None):
        ok = False
        for rec in self.recorders:
            try:
                can_record = rec.can_record() if hasattr(rec, 'can_record') else bool(getattr(rec, 'enabled', False))
                if getattr(rec, 'enabled', False) and can_record:
                    ok = bool(rec.start(session_id=session_id)) or ok
            except Exception as exc:
                print(f'[RAW] recorder start failed camera={self.camera_id}: {type(exc).__name__}: {exc}', flush=True)
        return ok

    def stop(self):
        ok = False
        for rec in self.recorders:
            try:
                ok = bool(rec.stop()) or ok
            except Exception as exc:
                print(f'[RAW] recorder stop failed camera={self.camera_id}: {type(exc).__name__}: {exc}', flush=True)
        return ok

    def toggle(self, session_id=None):
        return self.stop() if self.is_recording() else self.start(session_id=session_id)

    def is_recording(self):
        for rec in self.recorders:
            try:
                if bool(rec.is_recording()):
                    return True
            except Exception:
                pass
        return False

    def submit(self, evs, ts_us: int, wall_us: int, sync_meta: Optional[Dict[str, Any]] = None):
        submitted = False
        for rec in self.recorders:
            submit = getattr(rec, 'submit', None)
            if callable(submit):
                submitted = bool(submit(evs, ts_us, wall_us, sync_meta=sync_meta)) or submitted
        return submitted

    def status(self):
        details: Dict[str, Any] = {}
        for rec in self.recorders:
            name = str(getattr(rec, '__class__', type(rec)).__name__)
            try:
                if hasattr(rec, 'status'):
                    details[name] = rec.status()
                else:
                    details[name] = {
                        'enabled': bool(getattr(rec, 'enabled', False)),
                        'recording': bool(rec.is_recording()) if hasattr(rec, 'is_recording') else False,
                        'path': str(getattr(rec, 'current_path', '') or ''),
                    }
            except Exception as exc:
                details[name] = {'last_error': f'{type(exc).__name__}: {exc}'}
        event_pack = details.get('EventSlicePackRecorder', {})
        return {
            'enabled': bool(self.enabled),
            'recording': bool(self.is_recording()),
            'backend': 'combined_event_raw',
            'event_slice_pack': event_pack,
            'recorders': details,
        }

    def close(self):
        for rec in self.recorders:
            try:
                rec.close()
            except Exception:
                pass


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
                              default="NoFilter",
                              choices=["NoFilter", "FilterNegative", "FilterPositive", "SeparateFilter"])
    algo_options.add_argument('--cell-width', dest='cell_width', type=int, default=5)
    algo_options.add_argument('--cell-height', dest='cell_height', type=int, default=5)
    algo_options.add_argument('--processing-accumulation-time', dest='accumulation_time_us', type=int, default=3000)
    algo_options.add_argument('--static-memory', dest='static_memory_us', type=int, default=1000000)
    algo_options.add_argument('--untracked-threshold', dest='untracked_ths', type=int, default=5)
    algo_options.add_argument('--min-track-time', dest='min_track_time', type=int, default=1000)
    algo_options.add_argument('-a', '--activation-threshold', dest='activation_ths', type=int, default=10)
    algo_options.add_argument('--apply-filter', dest='apply_filter', default=False, action='store_true')
    algo_options.add_argument('-D', '--max-distance', dest='max_distance', type=int, default=80)
    algo_options.add_argument('--max-size-variation', dest='max_size_variation', type=int, default=100)
    algo_options.add_argument('--min-dist-moving-obj', dest='min_dist_moving_obj', type=int, default=3)
    algo_options.add_argument('--min-size', dest='min_size', type=int, default=5)
    algo_options.add_argument('--max-size', dest='max_size', type=int, default=100)
    algo_options.add_argument('--nozone', nargs='+', action='append', type=int,
                              metavar='center_x center_y radius inside', default=[])

    hot_pixel_options = parser.add_argument_group('Event hot-pixel / cluster diagnostics')
    hot_pixel_options.add_argument('--event-hot-pixel-filter-enable', dest='event_hot_pixel_filter_enable',
                                   action='store_true', default=False,
                                   help='Enable tracking-only hot-pixel suppression before STC/Spatter.')
    hot_pixel_options.add_argument('--disable-event-hot-pixel-filter', dest='event_hot_pixel_filter_enable',
                                   action='store_false')
    hot_pixel_options.add_argument('--event-hot-pixel-filter-mode', dest='event_hot_pixel_filter_mode',
                                   type=str, default='tracking_only')
    hot_pixel_options.add_argument('--event-hot-pixel-radius-px', dest='event_hot_pixel_radius_px',
                                   type=int, default=0)
    hot_pixel_options.add_argument('--event-hot-pixel-static-map', dest='event_hot_pixel_static_map',
                                   type=str, default='',
                                   help='Static hot pixels, e.g. "E:228,19;F:202,272" or JSON dict.')
    hot_pixel_options.add_argument('--event-hot-pixel-learn-enable', dest='event_hot_pixel_learn_enable',
                                   action='store_true', default=False)
    hot_pixel_options.add_argument('--disable-event-hot-pixel-learn', dest='event_hot_pixel_learn_enable',
                                   action='store_false')
    hot_pixel_options.add_argument('--event-hot-pixel-learn-min-fraction', dest='event_hot_pixel_learn_min_fraction',
                                   type=float, default=0.25)
    hot_pixel_options.add_argument('--event-hot-pixel-learn-min-count', dest='event_hot_pixel_learn_min_count',
                                   type=int, default=64)
    hot_pixel_options.add_argument('--event-hot-pixel-learn-confirm-slices',
                                   dest='event_hot_pixel_learn_confirm_slices', type=int, default=5)
    hot_pixel_options.add_argument('--event-hot-pixel-learn-max-pixels',
                                   dest='event_hot_pixel_learn_max_pixels', type=int, default=16)
    hot_pixel_options.add_argument('--event-cluster-sanitize-enable', dest='event_cluster_sanitize_enable',
                                   action='store_true', default=True)
    hot_pixel_options.add_argument('--disable-event-cluster-sanitize', dest='event_cluster_sanitize_enable',
                                   action='store_false')

    line_options = parser.add_argument_group('Realtime line filter options')
    line_options.add_argument('--use-cpp-line-filter', dest='use_cpp_line_filter', action='store_true', default=False,
                              help='使用全手写 C++ CppLinearMotionFilter 最终实现替代 Python LinearMotionFilter；检测流程/输出接口保持一致。')
    line_options.add_argument('--disable-cpp-line-filter', dest='use_cpp_line_filter', action='store_false',
                              help='强制使用 Python 原版 LinearMotionFilter。')
    line_options.add_argument('--line-history-len', dest='line_history_len', type=int, default=8)
    line_options.add_argument('--line-draw-history-len', dest='line_draw_history_len', type=int, default=40)
    line_options.add_argument('--line-pre-confirm-history-len', dest='line_pre_confirm_history_len', type=int,
                              default=24)
    line_options.add_argument('--line-draw-hold-missed-frames', dest='line_draw_hold_missed_frames', type=int,
                              default=2)
    line_options.add_argument('--line-draw-connect-max-gap-px', dest='line_draw_connect_max_gap_px', type=float,
                              default=72.0)
    line_options.add_argument('--disable-line-draw-kinematic-fit', dest='line_draw_kinematic_fit_enabled',
                              action='store_false', default=True,
                              help='禁用 P7 运动学拟合绘制，退回直接连接检测框中心点')
    line_options.add_argument('--line-draw-kinematic-residual-px', dest='line_draw_kinematic_residual_px', type=float,
                              default=0.0,
                              help='P7 运动学拟合残差阈值 px；0=自动，越小越不相信偏离点')
    line_options.add_argument('--line-draw-kinematic-max-curve-px', dest='line_draw_kinematic_max_curve_px', type=float,
                              default=32.0,
                              help='P7 允许的最大曲线弯曲幅度 px，越小越接近直线')
    line_options.add_argument('--line-draw-kinematic-max-accel', dest='line_draw_kinematic_max_accel_px_per_ms2',
                              type=float, default=0.18,
                              help='P7 允许的最大拟合加速度 px/ms^2，超出则退回直线')
    line_options.add_argument('--line-draw-kinematic-sample-step-px', dest='line_draw_kinematic_sample_step_px',
                              type=float, default=7.0,
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
    line_options.add_argument('--model-weak-correction-gain', dest='model_weak_correction_gain', type=float,
                              default=0.10,
                              help='P8 可疑观测对模型的弱校正权重')
    line_options.add_argument('--model-max-accel', dest='model_max_accel_px_ms2', type=float, default=0.10,
                              help='P9 模型速度变化上限 px/ms^2，防止单点观测把轨迹拉弯')
    line_options.add_argument('--model-outlier-kill-enabled', dest='model_outlier_kill_enabled', action='store_true',
                              default=True,
                              help='启用 P9 大残差立即终止；JSON 正向开关兼容，默认启用')
    line_options.add_argument('--disable-model-outlier-kill', dest='model_outlier_kill_enabled', action='store_false',
                              default=True,
                              help='禁用 P9 大残差立即终止；默认启用，防止轨迹穿过人物区域')
    line_options.add_argument('--model-outlier-kill-frames', dest='model_outlier_kill_frames', type=int, default=1,
                              help='连续多少帧模型大残差后终止，默认 1')
    line_options.add_argument('--raw-id-sticky-guard-enabled', dest='raw_id_sticky_guard_enabled', action='store_true',
                              default=True,
                              help='P13 启用 raw_id 粘背景保护，防止 Spatter raw_id 停在黑色障碍物上后把子弹轨迹拉断；默认启用')
    line_options.add_argument('--disable-raw-id-sticky-guard', dest='raw_id_sticky_guard_enabled', action='store_false',
                              default=True,
                              help='禁用 P13 raw_id 粘背景保护')
    line_options.add_argument('--raw-id-sticky-guard-min-speed', dest='raw_id_sticky_guard_min_speed_px_ms', type=float,
                              default=0.35,
                              help='P13 raw_id 粘背景保护启用的最低模型速度 px/ms')
    line_options.add_argument('--raw-id-sticky-guard-static-step-px', dest='raw_id_sticky_guard_static_step_px',
                              type=float, default=2.2,
                              help='P13 已确认子弹 raw_id 直连时，低于该步长视为疑似粘背景')
    line_options.add_argument('--raw-id-sticky-guard-residual-px', dest='raw_id_sticky_guard_residual_px', type=float,
                              default=24.0,
                              help='P13 raw_id 直连点距离模型预测超过该值时不再直接绑定旧弹道')
    line_options.add_argument('--bullet-id-stitch-enabled', dest='bullet_id_stitch_enabled', action='store_true',
                              default=True,
                              help='P16 启用已确认子弹段 ID stitching；新 raw_id 若几何上是旧 bullet 的延续，则继承旧 bullet_id/display_id')
    line_options.add_argument('--disable-bullet-id-stitch', dest='bullet_id_stitch_enabled', action='store_false',
                              default=True,
                              help='禁用 P16 bullet_id stitching')
    line_options.add_argument('--bullet-id-stitch-window-ms', dest='bullet_id_stitch_window_ms', type=float,
                              default=120.0,
                              help='P16 已确认 bullet 段续接最大时间窗口 ms')
    line_options.add_argument('--bullet-id-stitch-corridor-px', dest='bullet_id_stitch_corridor_px', type=float,
                              default=24.0,
                              help='P16 新段到旧弹道方向走廊的最大横向距离 px')
    line_options.add_argument('--bullet-id-stitch-min-dir-cos', dest='bullet_id_stitch_min_dir_cos', type=float,
                              default=0.82,
                              help='P16 新段方向和旧弹道方向的最小 cos 值')
    line_options.add_argument('--bullet-id-stitch-min-speed', dest='bullet_id_stitch_min_speed_px_ms', type=float,
                              default=0.32,
                              help='P16 续接所需最小局部/隐含速度 px/ms')
    line_options.add_argument('--bullet-id-stitch-max-size-ratio', dest='bullet_id_stitch_max_size_ratio', type=float,
                              default=5.5,
                              help='P16 续接允许的新旧 bbox 最大尺寸比例')
    line_options.add_argument('--bullet-id-stitch-reverse-enabled', dest='bullet_id_stitch_reverse_enabled',
                              action='store_true', default=True,
                              help='P17 启用反向直线 ID stitching；新段沿自身直线反向查找旧 bullet_id，补偿遮挡/旧 raw_id 粘背景')
    line_options.add_argument('--disable-bullet-id-stitch-reverse', dest='bullet_id_stitch_reverse_enabled',
                              action='store_false', default=True,
                              help='禁用 P17 反向直线 ID stitching')
    line_options.add_argument('--bullet-id-stitch-reverse-window-ms', dest='bullet_id_stitch_reverse_window_ms',
                              type=float, default=170.0,
                              help='P17 反向直线查找旧 bullet_id 的最大时间窗口 ms')
    line_options.add_argument('--bullet-id-stitch-reverse-corridor-px', dest='bullet_id_stitch_reverse_corridor_px',
                              type=float, default=26.0,
                              help='P17 旧 bullet 到新段反向延长线的最大横向距离 px')
    line_options.add_argument('--bullet-id-stitch-reverse-min-dir-cos', dest='bullet_id_stitch_reverse_min_dir_cos',
                              type=float, default=0.82,
                              help='P17 新段方向和旧 bullet 方向的最小 cos 值')
    line_options.add_argument('--bullet-id-stitch-reverse-min-local-disp-px',
                              dest='bullet_id_stitch_reverse_min_local_disp_px', type=float, default=10.0,
                              help='P17 允许反向续接前，新段自身至少需要的局部位移 px')
    line_options.add_argument('--bullet-id-stitch-reverse-max-size-ratio',
                              dest='bullet_id_stitch_reverse_max_size_ratio', type=float, default=5.8,
                              help='P17 反向续接允许的新旧 bbox 最大尺寸比例')
    line_options.add_argument('--bullet-id-bounce-link-enabled', dest='bullet_id_bounce_link_enabled',
                              action='store_true', default=True,
                              help='P19 启用 C++ 物理子弹反弹续接：反弹前后段继承同一 bullet_id/shot，但不直接判 HIT')
    line_options.add_argument('--disable-bullet-id-bounce-link', dest='bullet_id_bounce_link_enabled',
                              action='store_false', default=True,
                              help='禁用 P19 反弹续接')
    line_options.add_argument('--bullet-id-bounce-link-window-ms', dest='bullet_id_bounce_link_window_ms', type=float,
                              default=260.0,
                              help='P19 反弹续接最大时间窗口 ms')
    line_options.add_argument('--bullet-id-bounce-link-max-gap-px', dest='bullet_id_bounce_link_max_gap_px', type=float,
                              default=95.0,
                              help='P19 反弹续接基础最大空间间隔 px')
    line_options.add_argument('--bullet-id-bounce-link-adaptive-max-px', dest='bullet_id_bounce_link_adaptive_max_px',
                              type=float, default=190.0,
                              help='P19 反弹续接速度自适应最大空间间隔 px')
    line_options.add_argument('--bullet-id-bounce-link-min-turn-angle-deg',
                              dest='bullet_id_bounce_link_min_turn_angle_deg', type=float, default=95.0,
                              help='P19 反弹续接最小方向变化角度 deg')
    line_options.add_argument('--bullet-id-bounce-link-min-speed', dest='bullet_id_bounce_link_min_speed_px_ms',
                              type=float, default=0.38,
                              help='P19 反弹续接最低前段/后段/隐含速度 px/ms')
    line_options.add_argument('--bullet-id-bounce-link-max-size-ratio', dest='bullet_id_bounce_link_max_size_ratio',
                              type=float, default=6.5,
                              help='P19 反弹续接允许的新旧 bbox 最大尺寸比例')
    line_options.add_argument('--bullet-id-bounce-link-candidate-max-points',
                              dest='bullet_id_bounce_link_candidate_max_points', type=int, default=12,
                              help='P19 新段最多积累多少点仍允许反弹继承旧 bullet_id')
    line_options.add_argument('--line-draw-kinematic-force-straight-line',
                              dest='line_draw_kinematic_force_straight_line', action='store_true', default=True,
                              help='启用 P9 分段直线绘制；JSON 正向开关兼容，默认启用')
    line_options.add_argument('--disable-line-draw-force-straight', dest='line_draw_kinematic_force_straight_line',
                              action='store_false', default=True,
                              help='禁用 P9 分段直线绘制；默认强制单段直线，不拟合二次曲线')
    line_options.add_argument('--line-draw-after-terminate-hold-ms', dest='line_draw_after_terminate_hold_ms', type=int,
                              default=0,
                              help='终止后额外保留轨迹显示多久；P9 默认 0，即关闭飞行残留')
    line_options.add_argument('--line-draw-extend-tail-after-terminate', dest='line_draw_extend_tail_after_terminate',
                              action='store_true', default=False,
                              help='启用终止后速度外推尾迹；JSON 正向开关兼容，P9 默认关闭')
    line_options.add_argument('--enable-line-draw-extend-tail-after-terminate',
                              dest='line_draw_extend_tail_after_terminate', action='store_true', default=False,
                              help='启用终止后速度外推尾迹；P9 默认关闭')
    line_options.add_argument('--line-bootstrap-frames', dest='line_bootstrap_frames', type=int, default=0)
    line_options.add_argument('--line-angle-thresh-deg', dest='line_angle_thresh_deg', type=float, default=16.0)
    line_options.add_argument('--line-max-offset-px', dest='line_max_offset_px', type=float, default=4.0)
    line_options.add_argument('--line-min-step-px', dest='line_min_step_px', type=float, default=3.0)
    line_options.add_argument('--line-min-total-disp-px', dest='line_min_total_disp_px', type=float, default=20.0)
    line_options.add_argument('--line-max-missed-frames', dest='line_max_missed_frames', type=int, default=4)
    line_options.add_argument('--line-same-direction-ratio-thresh', dest='line_same_direction_ratio_thresh', type=float,
                              default=0.75)
    line_options.add_argument('--line-min-valid-step-count', dest='line_min_valid_step_count', type=int, default=2)
    line_options.add_argument('--line-min-valid-step-ratio', dest='line_min_valid_step_ratio', type=float, default=0.55)
    line_options.add_argument('--bullet-min-output-streak', dest='bullet_min_output_streak', type=int, default=5)
    line_options.add_argument('--event-line-shadow-enable', dest='event_line_shadow_enable',
                              action='store_true', default=False,
                              help='Run a per-camera shadow LineFilter for diagnostics only; does not emit points.')
    line_options.add_argument('--disable-event-line-shadow', dest='event_line_shadow_enable',
                              action='store_false')
    line_options.add_argument('--event-line-shadow-cameras', dest='event_line_shadow_cameras',
                              type=str, default='', help='Comma-separated camera aliases for shadow LineFilter.')
    line_options.add_argument('--event-line-shadow-min-step-px', dest='event_line_shadow_min_step_px',
                              type=float, default=-1.0)
    line_options.add_argument('--event-line-shadow-min-total-disp-px', dest='event_line_shadow_min_total_disp_px',
                              type=float, default=-1.0)
    line_options.add_argument('--event-line-shadow-bullet-min-output-streak',
                              dest='event_line_shadow_bullet_min_output_streak', type=int, default=-1)
    line_options.add_argument('--event-line-shadow-untracked-threshold',
                              dest='event_line_shadow_untracked_threshold', type=int, default=-1)
    line_options.add_argument('--event-line-shadow-line-min-valid-step-count',
                              dest='event_line_shadow_line_min_valid_step_count', type=int, default=-1)
    line_options.add_argument('--assoc-max-distance-px', dest='assoc_max_distance_px', type=float, default=36.0)
    line_options.add_argument('--assoc-direction-penalty-px', dest='assoc_direction_penalty_px', type=float,
                              default=10.0)
    line_options.add_argument('--assoc-max-size-ratio', dest='assoc_max_size_ratio', type=float, default=3.0)
    # P0/P1 新增参数
    line_options.add_argument('--maintain-max-offset-px', dest='maintain_max_offset_px', type=float, default=5.0,
                              help='maintain 阶段 max_offset 超此值立即 kill (default: 5.0)')
    line_options.add_argument('--maintain-max-static-frames', dest='maintain_max_static_frames', type=int, default=5,
                              help='maintain 阶段 static_fail_count 超此值 kill (default: 5, 原18)')
    line_options.add_argument('--maintain-max-bbox-std', dest='maintain_max_bbox_std', type=float, default=4.5,
                              help='maintain 阶段 bbox 宽高 std 超此值 kill (default: 4.5)')
    line_options.add_argument('--probation-max-offset-px', dest='probation_max_offset_px', type=float, default=4.5,
                              help='probation 阶段 max_offset 上限 (default: 4.5)')
    line_options.add_argument('--disable-ghost-capture', dest='ghost_capture_enabled', action='store_false',
                              default=True,
                              help='禁用 ghost 跨 track 捕获')
    # P2-A: recent offset 连续超限快速 kill
    line_options.add_argument('--maintain-recent-offset-px', dest='maintain_recent_offset_px', type=float, default=4.0,
                              help='maintain 阶段 recent_max_offset 超此值连续 K 帧则 kill (default: 4.0)')
    line_options.add_argument('--maintain-recent-offset-exceed-frames', dest='maintain_recent_offset_exceed_frames',
                              type=int, default=3,
                              help='recent_offset 超限连续帧数阈值 (default: 3)')
    # P2-B: bbox EMA 慢漂移检测
    line_options.add_argument('--maintain-bbox-ema-alpha', dest='maintain_bbox_ema_alpha', type=float, default=0.25,
                              help='bbox EMA 平滑系数，越小越平滑 (default: 0.25)')
    line_options.add_argument('--maintain-bbox-ema-drift-ratio', dest='maintain_bbox_ema_drift_ratio', type=float,
                              default=0.45,
                              help='当前帧 bbox 相对 EMA 均值的最大允许偏差率 (default: 0.45)')
    # P2-C: maintain_turn 次数和累计角度限制
    line_options.add_argument('--maintain-max-turn-count', dest='maintain_max_turn_count', type=int, default=2,
                              help='允许的最大 maintain_turn 次数，超过则 kill (default: 2)')
    line_options.add_argument('--maintain-max-cumulative-turn-deg', dest='maintain_max_cumulative_turn_deg', type=float,
                              default=90.0,
                              help='允许的最大累计转向角度（度），超过则 kill (default: 90.0)')
    # P2-E: 反弹续接距离
    line_options.add_argument('--ghost-bounce-max-dist-px', dest='ghost_bounce_max_dist_px', type=float, default=35.0,
                              help='反弹续接分支：新点到旧终止点的最大允许距离（像素）(default: 35.0，P3缩小)')
    line_options.add_argument('--ghost-bounce-min-speed', dest='ghost_bounce_min_speed_px_per_ms', type=float,
                              default=0.60,
                              help='反弹续接最低隐含速度 px/ms，过低视为人体静止重现 (default: 0.60)')
    line_options.add_argument('--same-track-bbox-reuse-min-disp-px', dest='same_track_bbox_reuse_min_disp_px',
                              type=float, default=6.0,
                              help='terminated_bbox / terminated_bbox_ema 后允许 same-track 重用前，当前点相对终止点至少要新增的位移 (default: 6.0)')
    line_options.add_argument('--trigger-max-full-offset-ratio', dest='trigger_max_full_offset_ratio', type=float,
                              default=1.5,
                              help='trigger 阶段 full_offset 上限倍率（相对 max_line_offset_px）(default: 1.5)')
    line_options.add_argument('--trigger-max-sign-flip-ratio', dest='trigger_max_sign_flip_ratio', type=float,
                              default=0.35,
                              help='trigger 阶段相邻步方向反转比率上限 (default: 0.35)')
    line_options.add_argument('--trigger-min-avg-speed', dest='trigger_min_avg_speed_px_per_ms', type=float,
                              default=0.60,
                              help='trigger 阶段平均速度下限 px/ms，低于此值视为慢漂移误判 (default: 0.60)')
    line_options.add_argument('--trigger-raw-min-valid-steps', dest='trigger_raw_min_valid_steps', type=int, default=3,
                              help='raw_id 新生 trigger 的最小 recent valid_step_count (default: 3)')
    line_options.add_argument('--trigger-sparse-burst-max-valid-steps', dest='trigger_sparse_burst_max_valid_steps',
                              type=int, default=2,
                              help='稀疏大跳 veto 的最大 recent valid_step_count (default: 2)')
    line_options.add_argument('--trigger-sparse-burst-min-disp-per-step-px',
                              dest='trigger_sparse_burst_min_disp_per_step_px', type=float, default=18.0,
                              help='稀疏大跳 veto 的最小每步位移 px/step (default: 18.0)')
    line_options.add_argument('--trigger-sparse-burst-min-total-disp-px', dest='trigger_sparse_burst_min_total_disp_px',
                              type=float, default=35.0,
                              help='稀疏大跳 veto 的最小 recent 总位移 (default: 35.0)')
    line_options.add_argument('--trigger-sparse-burst-max-path-over-net', dest='trigger_sparse_burst_max_path_over_net',
                              type=float, default=1.03,
                              help='稀疏大跳 veto 的最大 recent path_over_net (default: 1.03)')
    line_options.add_argument('--trigger-old-raw-age-ms', dest='trigger_old_raw_age_ms', type=float, default=120.0,
                              help='老 raw 轨迹触发更严门槛的年龄阈值 ms (default: 120)')
    line_options.add_argument('--trigger-old-raw-min-valid-steps', dest='trigger_old_raw_min_valid_steps', type=int,
                              default=4,
                              help='老 raw 轨迹 trigger 的最小 recent valid_step_count (default: 4)')
    line_options.add_argument('--probation-new-min-extra-disp-px', dest='probation_new_min_extra_disp_px', type=float,
                              default=8.0,
                              help='new/raw 出生后 probation 期间要求的最小新增位移 (default: 8.0)')
    line_options.add_argument('--probation-new-min-extra-valid-steps', dest='probation_new_min_extra_valid_steps',
                              type=int, default=2,
                              help='new/raw 出生后 probation 期间要求的最小新增有效步数 (default: 2)')
    line_options.add_argument('--debug-draw-raw-clusters', dest='debug_draw_raw_clusters', action='store_true',
                              default=False,
                              help='调试用：绘制 SpatterTracker 原始 cluster 框；默认关闭，避免人多时本地窗口绘制过重。')
    line_options.add_argument('--disable-debug-draw-raw-clusters', dest='debug_draw_raw_clusters', action='store_false',
                              help='显式关闭原始 cluster 框绘制。')
    line_options.add_argument('--csvdebug-track', dest='debug_track_csv', type=str, default='')
    line_options.add_argument('--spatter-raw-debug-csv', dest='spatter_raw_debug_csv', type=str, default='',
                              help='诊断用：逐 slice 保存 SpatterTracker 原始 cluster，判断飞溅物层是否先断。')
    line_options.add_argument('--spatter-summary-print-ms', dest='spatter_summary_print_ms', type=int, default=0,
                              help='诊断用：每隔 N ms 打印 Spatter 原始 cluster/accepted 统计；0=关闭。')

    tsf1_pub_options = parser.add_argument_group('TSF1 publish options')
    tsf1_pub_options.add_argument('--enable-tsf1-pub', dest='enable_tsf1_pub', default=False, action='store_true')
    tsf1_pub_options.add_argument('--tsf1-pub-name', dest='tsf1_pub_name', type=str, default='bullet_ts_latest')
    tsf1_pub_options.add_argument('--tsf1-pub-scale', dest='tsf1_pub_scale', type=float, default=0.7)
    tsf1_pub_options.add_argument('--tsf1-pub-fps', dest='tsf1_pub_fps', type=float, default=8.0)
    tsf1_pub_options.add_argument('--tsf1-pub-accumulation-time-us', dest='tsf1_pub_accumulation_time_us', type=int,
                                  default=12000)
    tsf1_pub_options.add_argument('--tsf1-pub-alpha', dest='tsf1_pub_alpha', type=float, default=1.8)
    tsf1_pub_options.add_argument('--tsf1-pub-beta', dest='tsf1_pub_beta', type=float, default=0.0)
    tsf1_pub_options.add_argument('--tsf1-pub-dilate-ksize', dest='tsf1_pub_dilate_ksize', type=int, default=3)
    tsf1_pub_options.add_argument('--tsf1-pub-dilate-iters', dest='tsf1_pub_dilate_iters', type=int, default=1)
    tsf1_pub_options.add_argument('--tsf1-server-ip', dest='tsf1_server_ip', type=str, default='')
    tsf1_pub_options.add_argument('--tsf1-server-port', dest='tsf1_server_port', type=int, default=5001)
    tsf1_pub_options.add_argument('--tsf1-camera-id', dest='tsf1_camera_id', type=str, default='cam0')
    tsf1_pub_options.add_argument('--tsf1-codec', dest='tsf1_codec', type=str, default='zlib', choices=['raw', 'zlib'])
    tsf1_pub_options.add_argument('--tsf1-zlib-level', dest='tsf1_zlib_level', type=int, default=1)
    tsf1_pub_options.add_argument('--tsf1-ts-mode', dest='tsf1_ts_mode', type=str, default='linear',
                                  choices=['linear', 'exp'])
    tsf1_pub_options.add_argument('--tsf1-ts-decay-us', dest='tsf1_ts_decay_us', type=int, default=65000)
    tsf1_pub_options.add_argument('--tsf1-post-alpha', dest='tsf1_post_alpha', type=float, default=1.0)
    tsf1_pub_options.add_argument('--tsf1-post-beta', dest='tsf1_post_beta', type=float, default=0.0)
    tsf1_pub_options.add_argument('--tsf1-post-dilate-ksize', dest='tsf1_post_dilate_ksize', type=int, default=0)
    tsf1_pub_options.add_argument('--tsf1-post-dilate-iters', dest='tsf1_post_dilate_iters', type=int, default=0)
    tsf1_pub_options.add_argument('--tsf1-auto-start-sender', dest='tsf1_auto_start_sender', default=False,
                                  action='store_true')
    tsf1_pub_options.add_argument('--tsf1-sender-script', dest='tsf1_sender_script', type=str, default='')
    tsf1_pub_options.add_argument('--enable-event-overlay-pub', dest='enable_event_overlay_pub', default=False,
                                  action='store_true',
                                  help='发布事件相机黑底渲染画面到共享内存，供 MVS 端做融合预览')
    tsf1_pub_options.add_argument('--event-overlay-pub-name', dest='event_overlay_pub_name', type=str,
                                  default='event_overlay_latest')
    tsf1_pub_options.add_argument('--event-overlay-pub-scale', dest='event_overlay_pub_scale', type=float, default=1.0)
    tsf1_pub_options.add_argument('--event-overlay-pub-channels', dest='event_overlay_pub_channels', type=int,
                                  default=3, choices=[1, 3])
    tsf1_pub_options.add_argument('--event-overlay-pub-mode', dest='event_overlay_pub_mode', type=str,
                                  default='overlay',
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

    obs_options.add_argument('--obs-restart-max-consecutive-failures', dest='obs_restart_max_consecutive_failures', type=int, default=8,
                             help='OBS/SRT ffmpeg consecutive failure threshold before pausing restarts.')
    obs_options.add_argument('--obs-restart-circuit-breaker-sec', dest='obs_restart_circuit_breaker_sec', type=float, default=60.0,
                             help='OBS/SRT ffmpeg pause duration after consecutive restart failures.')
    obs_options.add_argument('--obs-restart-max-backoff-sec', dest='obs_restart_max_backoff_sec', type=float, default=30.0,
                             help='OBS/SRT ffmpeg restart backoff upper bound.')

    hit_options = parser.add_argument_group('Bullet hit event options')
    hit_options.add_argument('--enable-hit-events', dest='enable_hit_events', default=False, action='store_true',
                             help='启用子弹命中事件 UDP 发送')
    hit_options.add_argument('--hit-server-ip', dest='hit_server_ip', type=str, default='',
                             help='命中判定服务器 IP（通常与 TSF1 服务器相同）')
    hit_options.add_argument('--hit-server-port', dest='hit_server_port', type=int, default=5003,
                             help='命中判定服务器 UDP 端口（默认 5003）')
    hit_options.add_argument('--hit-camera-alias', dest='hit_camera_alias', type=str, default='',
                             help='相机可选显示名，不影响逻辑，--tsf1-camera-id 填序列号即可')

    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument('-f', '--replay_factor', type=float, default=1)

    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument('-o', '--out-video', dest='out_video', type=str, default="")
    outcome_options.add_argument('--out-video-mode', dest='out_video_mode', type=str, default='overlay',
                                 choices=['overlay', 'raw', 'bullet'],
                                 help=(
                                     '本地保存视频画面模式：'
                                     'overlay=事件画面+子弹轨迹/ID/禁区；'
                                     'raw=只保存事件相机原始渲染画面；'
                                     'bullet=黑底只保存子弹轨迹/ID。'
                                 ))
    outcome_options.add_argument('-l', '--log-results', dest='out_log', type=str, default="")
    outcome_options.add_argument('--bullet-effect-log', dest='bullet_effect_log', type=str, default="",
                                 help='标准子弹特效轨迹 CSV。会自动按相机 alias/SN 追加后缀，避免多相机写同一个文件。')

    hw_group = parser.add_argument_group('Live camera HAL tuning')
    hw_group.add_argument('--enable-digital-crop', action='store_true', default=False)
    hw_group.add_argument('--disable-digital-crop', action='store_true', default=False)
    hw_group.add_argument('--crop-region', nargs=4, type=int, metavar=('START_X', 'START_Y', 'END_X', 'END_Y'),
                          default=None)
    hw_group.add_argument('--crop-reset-origin', action='store_true', default=False)
    hw_group.add_argument('--enable-afk', action='store_true', default=False)
    hw_group.add_argument('--disable-afk', action='store_true', default=False)
    hw_group.add_argument('--afk-low-freq', type=int, default=0)
    hw_group.add_argument('--afk-high-freq', type=int, default=0)
    hw_group.add_argument('--afk-mode', choices=['band_stop', 'band_pass'], default='band_stop')
    hw_group.add_argument('--afk-duty-cycle', type=float, default=None)
    hw_group.add_argument('--afk-start-threshold', type=int, default=None)
    hw_group.add_argument('--afk-stop-threshold', type=int, default=None)
    hw_group.add_argument('--enable-hw-trail', action='store_true', default=False)
    hw_group.add_argument('--disable-hw-trail', action='store_true', default=False)
    hw_group.add_argument('--hw-trail-type', choices=['trail', 'stc_cut_trail', 'stc_keep_trail'],
                          default='stc_cut_trail')
    hw_group.add_argument('--hw-trail-threshold-us', type=int, default=0)
    hw_group.add_argument('--enable-erc', action='store_true', default=False)
    hw_group.add_argument('--disable-erc', action='store_true', default=False)
    hw_group.add_argument('--erc-cd-event-rate', type=int, default=None)
    hw_group.add_argument('--set-bias', dest='bias_updates', action='append', default=[], metavar='NAME=VALUE')
    hw_group.add_argument('--print-biases', dest='print_biases', action='store_true', default=False)

    native_hal_group = parser.add_argument_group('Native HAL pipe bridge options')
    native_hal_group.add_argument('--event-native-hal-bridge-enable', dest='event_native_hal_bridge_enable',
                                  action='store_true', default=False,
                                  help='V6-B: use the C++ HAL DeviceDiscovery/raw pipe bridge instead of Python EventsIterator.')
    native_hal_group.add_argument('--disable-event-native-hal-bridge', dest='event_native_hal_bridge_enable',
                                  action='store_false',
                                  help='Disable the V6-B C++ HAL pipe bridge.')
    native_hal_group.add_argument('--event-native-hal-bridge-bin', dest='event_native_hal_bridge_bin',
                                  type=str, default='',
                                  help='Path to native_event_bridge/build/hal_event_pipe_bridge.')
    native_hal_group.add_argument('--event-native-hal-bridge-wait-mode', dest='event_native_hal_bridge_wait_mode',
                                  type=str, choices=['blocking', 'poll'], default='poll',
                                  help='HAL stream wait mode used by the C++ bridge.')
    native_hal_group.add_argument('--event-native-hal-bridge-poll-sleep-us',
                                  dest='event_native_hal_bridge_poll_sleep_us',
                                  type=int, default=100,
                                  help='Sleep used after empty HAL poll_buffer results.')
    native_hal_group.add_argument('--event-native-hal-bridge-stderr-log',
                                  dest='event_native_hal_bridge_stderr_log',
                                  type=str, default='./sync_ipc/event_hal_bridge_{camera}.log',
                                  help='Bridge stderr log path; supports {camera} and {serial}.')
    native_hal_group.add_argument('--event-native-hal-fifo-path',
                                  dest='event_native_hal_fifo_path',
                                  type=str, default='',
                                  help='V6-C: read event slices from a hal_event_fifo_broker FIFO instead of opening the camera in this worker.')

    throttle_options = parser.add_argument_group('Adaptive Throttling options')
    throttle_options.add_argument('--max-events-per-slice', dest='max_events_per_slice', type=int, default=30000,
                                  help='Max events per accumulation slice before throttling kicks in.')
    throttle_options.add_argument('--severe-events-per-slice', dest='severe_events_per_slice', type=int, default=60000,
                                  help='Threshold for severe throttling (more aggressive sampling).')
    throttle_options.add_argument('--burst-hold-ms', dest='burst_hold_ms', type=int, default=1500,
                                  help='Time in ms to wait after rate drops before recovering from throttling.')

    debug_options = parser.add_argument_group('Diagnostics options')
    debug_options.add_argument('--print-effective-args', dest='print_effective_args', action='store_true')
    debug_options.add_argument('--strict-line-filter-kwargs', dest='strict_line_filter_kwargs', action='store_true')
    debug_options.add_argument('--print-event-stats', dest='print_event_stats', action='store_true', default=False,
                               help='只打印代码实际收到的事件统计，不改变算法行为。')
    debug_options.add_argument('--event-stats-interval-ms', dest='event_stats_interval_ms', type=int, default=500,
                               help='事件统计打印间隔（毫秒）。')
    debug_options.add_argument('--enable-shake-guard', dest='enable_shake_guard', action='store_true', default=False,
                               help='启用相机抖动保护：触发时直接跳过重检测流程。')
    debug_options.add_argument('--shake-guard-rate-thresh-mevs', dest='shake_guard_rate_thresh_mevs', type=float,
                               default=14.0,
                               help='raw 事件率阈值（MEv/s），超过则认为进入抖动保护。')
    debug_options.add_argument('--shake-guard-trigger-slices', dest='shake_guard_trigger_slices', type=int, default=2,
                               help='连续多少个 slice 超阈值后触发抖动保护。')
    debug_options.add_argument('--shake-guard-hold-ms', dest='shake_guard_hold_ms', type=int, default=800,
                               help='触发后至少保持多长时间（毫秒）。')
    debug_options.add_argument('--shake-guard-release-ratio', dest='shake_guard_release_ratio', type=float,
                               default=0.85,
                               help='退出阈值比例：rate <= thresh * ratio 才允许退出。')
    debug_options.add_argument('--shake-guard-skip-tsf1', dest='shake_guard_skip_tsf1', action='store_true',
                               default=False,
                               help='抖动保护期间同时暂停 TSF1 发布，进一步减负。')
    debug_options.add_argument('--shake-guard-cluster-thresh', dest='shake_guard_cluster_thresh', type=int, default=0,
                               help='cluster 数量保护阈值；>0 时，当前帧 cluster 数超过该值会进入抖动保护，适合验证相机晃动导致 cluster 爆炸的问题。')
    debug_options.add_argument('--disable-spatter-processing', dest='disable_spatter_processing', action='store_true',
                               default=False,
                               help='调试用：跳过飞溅物检测、子弹轨迹过滤和子弹事件发送，只保留 RAW 录制、事件统计、TSF1/画面生成。')
    debug_options.add_argument('--enable-pre-spatter-guard', dest='enable_pre_spatter_guard', action='store_true',
                               default=False,
                               help='启用前置重载保护：在 spatter_tracker.process_events 之前根据事件数/事件率跳过重检测，防止实时窗口积压。')
    debug_options.add_argument('--pre-spatter-max-events-per-slice', dest='pre_spatter_max_events_per_slice', type=int,
                               default=18000,
                               help='前置保护：单个 accumulation slice 的事件数超过该值时进入保护；0 表示不按事件数触发。')
    debug_options.add_argument('--pre-spatter-rate-thresh-mevs', dest='pre_spatter_rate_thresh_mevs', type=float,
                               default=8.0,
                               help='前置保护：raw 事件率超过该值(MEv/s)时进入保护；0 表示不按事件率触发。')
    debug_options.add_argument('--pre-spatter-trigger-slices', dest='pre_spatter_trigger_slices', type=int, default=1,
                               help='前置保护：连续多少个 slice 超阈值后触发，默认 1 表示立即保护。')
    debug_options.add_argument('--pre-spatter-hold-ms', dest='pre_spatter_hold_ms', type=int, default=600,
                               help='前置保护：触发后至少保持多久，期间只刷新轻量画面，不跑 spatter/line_filter。')
    debug_options.add_argument('--pre-spatter-release-ratio', dest='pre_spatter_release_ratio', type=float,
                               default=0.70,
                               help='前置保护：退出阈值比例，事件数和事件率都低于阈值*ratio 后才退出。')
    debug_options.add_argument('--pre-spatter-reset-on-enter', dest='pre_spatter_reset_on_enter', action='store_true',
                               default=True,
                               help='前置保护触发时重建 spatter_tracker/line_filter，清掉被人群/抖动污染的状态；默认启用。')
    debug_options.add_argument('--disable-pre-spatter-reset-on-enter', dest='pre_spatter_reset_on_enter',
                               action='store_false',
                               help='前置保护触发时不重建检测状态，仅跳过重检测。')
    debug_options.add_argument('--pre-spatter-skip-tsf1', dest='pre_spatter_skip_tsf1', action='store_true',
                               default=False,
                               help='前置保护期间也暂停 TSF1 发布，进一步减负；默认仍发布 TSF1。')
    debug_options.add_argument('--pre-spatter-skip-obs', dest='pre_spatter_skip_obs', action='store_true',
                               default=False,
                               help='前置保护期间暂停 OBS 提交，进一步减负；默认仍提交 OBS。')
    debug_options.add_argument('--raw-record-light-mode', dest='raw_record_light_mode', action='store_true',
                               default=False,
                               help='RAW 录制期间自动进入轻量模式：只刷新画面和录制，不跑 spatter/line_filter/CSV，避免录制真实 RAW 时卡顿。')
    debug_options.add_argument('--raw-record-light-skip-obs', dest='raw_record_light_skip_obs', action='store_true',
                               default=True,
                               help='RAW 录制轻量模式下暂停 OBS 提交；默认启用。')
    debug_options.add_argument('--enable-hotkey-raw-record', dest='enable_hotkey_raw_record', action='store_true',
                               default=False,
                               help='启用窗口热键 RAW 录制：按 R 开始，再按 R 结束。')
    debug_options.add_argument('--raw-record-dir', dest='raw_record_dir', type=str, default='./raw_captures',
                               help='热键录制 RAW 文件保存目录。')
    debug_options.add_argument('--raw-record-prefix', dest='raw_record_prefix', type=str, default='capture',
                               help='热键录制 RAW 文件名前缀。')
    debug_options.add_argument('--enable-remote-raw-record-control', dest='enable_remote_raw_record_control',
                               action='store_true', default=False,
                               help='启用 launcher 广播式多相机 RAW 录制控制。')
    debug_options.add_argument('--raw-record-command-file', dest='raw_record_command_file', type=str,
                               default='/tmp/eventcam_raw_record_cmd.json',
                               help='多相机广播录制控制命令文件。')
    debug_options.add_argument('--raw-record-command-poll-ms', dest='raw_record_command_poll_ms', type=int, default=50,
                               help='轮询广播录制命令文件的间隔（毫秒）。')

    debug_options.add_argument('--event-slice-record-enable', dest='event_slice_record_enable',
                               action='store_true', default=True,
                               help='Enable V0.2 event-slice replay-pack recording controlled by record_control.json.')
    debug_options.add_argument('--disable-event-slice-record', dest='event_slice_record_enable',
                               action='store_false',
                               help='Disable V0.2 event-slice replay-pack recording.')
    debug_options.add_argument('--event-slice-record-root', dest='event_slice_record_root', type=str,
                               default='./recordings',
                               help='Root directory for V0.2 EVENT_<camera> replay-pack files.')
    debug_options.add_argument('--event-slice-record-queue', dest='event_slice_record_queue', type=int,
                               default=512,
                               help='Per-event-worker queue depth for asynchronous event-slice recording.')
    debug_options.add_argument('--event-slice-record-flush-ms', dest='event_slice_record_flush_ms', type=int,
                               default=1000,
                               help='Flush interval for event-slice replay-pack manifest/index writes.')

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
    sync_options.add_argument('--ext-trigger-log-all-edges', dest='ext_trigger_log_all_edges',
                              action='store_true', default=False,
                              help='Debug only: write non-selected trigger edges to CSV. Default writes selected polarity only.')
    sync_options.add_argument('--ext-trigger-log-selected-only', dest='ext_trigger_log_all_edges',
                              action='store_false',
                              help='Write only selected polarity trigger edges to CSV to avoid huge sync logs.')
    sync_options.add_argument('--ext-trigger-print-every', dest='ext_trigger_print_every',
                              type=int, default=100,
                              help='每收到多少个有效同步 trigger 打印一次；0=不打印。')
    sync_options.add_argument('--ext-trigger-dedupe-ms', dest='ext_trigger_dedupe_ms',
                              type=float, default=18.0,
                              help='selected polarity trigger 去重窗口，单位 ms；40Hz 触发建议 18ms；0=关闭去重。')
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
                                    default='hard_events',
                                    choices=[
                                        'hard_events',
                                        'visual_only', 'visual-only',
                                        'stats_only', 'stats-only', 'observe', 'soft',
                                        'off', 'none', 'disabled', 'disable',
                                    ],
                                    help='人体 mask 过滤模式。hard_events=当前帧人体 mask 内事件直接不进入 spatter/line_filter。')
    human_mask_options.add_argument('--event-human-mask-jsonl-template', dest='event_human_mask_jsonl_template',
                                    type=str,
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
    human_mask_options.add_argument('--event-human-mask-wait-ms', dest='event_human_mask_wait_ms', type=float,
                                    default=0.0,
                                    help='每个 sync_index 首个事件 slice 等待当前帧人体结果的最长时间。0=不等待；若要严格用当前 MVS 帧覆盖 4ms slice，建议 40~60ms。')
    human_mask_options.add_argument('--event-human-mask-frame-hold-ms', dest='event_human_mask_frame_hold_ms',
                                    type=float, default=48.0,
                                    help='当前 MVS 帧人体 mask 的事件时间有效窗。事件 slice 只有在同 sync_index 且距离该 trigger 不超过此值时才使用该 mask；0=只按 sync_index，不按时间过期。')
    human_mask_options.add_argument('--event-human-mask-dilate-px', dest='event_human_mask_dilate_px', type=int,
                                    default=8,
                                    help='在事件相机坐标系下对人体 mask 膨胀的像素数，覆盖人体边缘抖动。')
    human_mask_options.add_argument('--event-human-mask-erode-px', dest='event_human_mask_erode_px', type=int,
                                    default=0,
                                    help='在事件相机坐标系下对人体 mask 腐蚀的像素数；移动误判优先时建议 0。')
    human_mask_options.add_argument('--event-human-mask-min-conf', dest='event_human_mask_min_conf', type=float,
                                    default=0.2,
                                    help='低于该 confidence 的人体不参与事件过滤。')
    human_mask_options.add_argument('--event-human-mask-max-records', dest='event_human_mask_max_records', type=int,
                                    default=256,
                                    help='每台事件相机缓存多少个 MVS 人体结果帧。')
    human_mask_options.add_argument('--event-human-mask-print-every-ms', dest='event_human_mask_print_every_ms',
                                    type=int, default=1000,
                                    help='事件人体 mask 过滤统计打印间隔；0=不打印。')
    human_mask_options.add_argument('--event-human-mask-use-carry-last', dest='event_human_mask_use_carry_last',
                                    action='store_true', default=False,
                                    help='允许 carry_last 人体结果参与事件 hard mask 过滤。默认关闭，因为 carry_last 是旧 mask，容易误删真实子弹事件。')
    human_mask_options.add_argument('--event-human-mask-stats-jsonl', dest='event_human_mask_stats_jsonl', type=str,
                                    default='',
                                    help='可选：把每个 event slice 的人体 hard mask 过滤统计写入 JSONL。支持 {camera}。')
    human_mask_options.add_argument('--event-human-mask-stats-max-fps', dest='event_human_mask_stats_max_fps',
                                    type=float, default=2.0,
                                    help='Limit human-mask diagnostic JSONL writes. 0 means unlimited; default 2Hz.')
    human_mask_options.add_argument('--event-human-mask-stats-flush-ms', dest='event_human_mask_stats_flush_ms',
                                    type=float, default=1000.0,
                                    help='Flush human-mask diagnostic JSONL at this interval. 0 flushes every write.')

    human_mask_options.add_argument('--event-human-mask-buffer-enable', dest='event_human_mask_buffer_enable',
                                    action='store_true', default=False,
                                    help='Enable sync-index event buffering before human-mask filtering.')
    human_mask_options.add_argument('--disable-event-human-mask-buffer', dest='event_human_mask_buffer_enable',
                                    action='store_false',
                                    help='Disable sync-index event buffering before human-mask filtering.')
    human_mask_options.add_argument('--event-human-mask-buffer-ms', dest='event_human_mask_buffer_ms',
                                    type=float, default=90.0,
                                    help='Nominal event hold budget in ms while waiting for the same-sync human mask.')
    human_mask_options.add_argument('--event-human-mask-buffer-timeout-ms', dest='event_human_mask_buffer_timeout_ms',
                                    type=float, default=150.0,
                                    help='Hard max event hold time in ms for raw-at-timeout policy.')
    human_mask_options.add_argument('--event-human-mask-buffer-release-policy',
                                    dest='event_human_mask_buffer_release_policy', type=str,
                                    default='raw_at_budget',
                                    choices=['raw_at_budget', 'raw_at_timeout', 'strict_same_sync_drop'],
                                    help='raw_at_budget releases unfiltered slices at buffer-ms; raw_at_timeout waits until timeout-ms; strict_same_sync_drop only releases same-sync masked slices and drops on timeout/pressure.')
    human_mask_options.add_argument('--event-human-mask-buffer-max-events',
                                    dest='event_human_mask_buffer_max_events', type=int, default=300000,
                                    help='Force-release oldest buffered event slice when total buffered events exceed this guard.')
    human_mask_options.add_argument('--event-human-mask-buffer-max-buckets',
                                    dest='event_human_mask_buffer_max_buckets', type=int, default=8,
                                    help='Force-release oldest sync bucket when buffered sync buckets exceed this guard.')
    human_mask_options.add_argument('--event-human-mask-buffer-empty-release-fps',
                                    dest='event_human_mask_buffer_empty_release_fps', type=float, default=0.0,
                                    help='Release empty buffered event slices at heartbeat FPS; 0=legacy immediate release.')
    human_mask_options.add_argument('--event-human-mask-filter-max-fps',
                                    dest='event_human_mask_filter_max_fps', type=float, default=0.0,
                                    help='When human-mask buffer is enabled, enforce at most this bucket/filter FPS; 0=buffer timing only.')
    human_mask_options.add_argument('--event-sync-source', dest='event_sync_source', type=str,
                                    default='event_local',
                                    choices=['event_local', 'mvs_soft_trigger', 'mvs_timeline', 'mvs_soft_trigger_timeline'],
                                    help='Event/MVS sync source. V9.1-R uses mvs_soft_trigger as the authoritative sync master.')
    human_mask_options.add_argument('--event-sync-timeline-jsonl', dest='event_sync_timeline_jsonl',
                                    type=str, default='./sync_ipc/global_trigger_timeline.jsonl',
                                    help='Authoritative MVS soft-trigger timeline JSONL.')
    human_mask_options.add_argument('--event-sync-map-jsonl-template', dest='event_sync_map_jsonl_template',
                                    type=str, default='./sync_ipc/event_trigger_map_{camera}.jsonl',
                                    help='Per-camera event-local to MVS-sync mapping JSONL template.')
    human_mask_options.add_argument('--event-sync-health-json-template', dest='event_sync_health_json_template',
                                    type=str, default='./sync_ipc/event_sync_health_{camera}.json',
                                    help='Per-camera Event/MVS sync health JSON template.')
    human_mask_options.add_argument('--event-sync-map-ring-json-template', dest='event_sync_map_ring_json_template',
                                    type=str, default='./sync_ipc/event_sync_map_ring_{camera}.json',
                                    help='Fixed-size per-camera Event/MVS sync map ring JSON for delayed BEV history selection.')
    human_mask_options.add_argument('--event-sync-map-ring-records', dest='event_sync_map_ring_records',
                                    type=int, default=512,
                                    help='Max records kept in event_sync_map_ring JSON.')
    human_mask_options.add_argument('--event-sync-map-ring-write-interval-ms',
                                    dest='event_sync_map_ring_write_interval_ms',
                                    type=float, default=100.0,
                                    help='Minimum wall interval for rewriting event_sync_map_ring JSON.')
    human_mask_options.add_argument('--event-sync-wall-guard-ms', dest='event_sync_wall_guard_ms',
                                    type=float, default=20.0,
                                    help='Max wall-time delta when locking an event trigger edge to a MVS soft-trigger tick.')
    human_mask_options.add_argument('--event-sync-offset-jump-warn-frames',
                                    dest='event_sync_offset_jump_warn_frames', type=int, default=2,
                                    help='Warn/degrade if event-local to MVS offset jumps by more than this many frames.')
    human_mask_options.add_argument('--event-sync-local-fallback-enable',
                                    dest='event_sync_local_fallback_enable', action='store_true', default=True,
                                    help='Allow sync_start tick_id + local trigger fallback before the MVS timeline is available.')
    human_mask_options.add_argument('--disable-event-sync-local-fallback',
                                    dest='event_sync_local_fallback_enable', action='store_false')
    human_mask_options.add_argument('--event-replay-mvs-pacing-enable',
                                    dest='event_replay_mvs_pacing_enable', action='store_true', default=True,
                                    help='In file replay direct-index mode, slow Event worker when it runs ahead of the MVS replay tick.')
    human_mask_options.add_argument('--disable-event-replay-mvs-pacing',
                                    dest='event_replay_mvs_pacing_enable', action='store_false')
    human_mask_options.add_argument('--event-replay-mvs-pacing-trigger-status-json',
                                    dest='event_replay_mvs_pacing_trigger_status_json',
                                    type=str, default='./sync_ipc/trigger_status.json',
                                    help='MVS replay trigger_status.json used as the pacing target.')
    human_mask_options.add_argument('--event-replay-mvs-pacing-lookahead-ticks',
                                    dest='event_replay_mvs_pacing_lookahead_ticks',
                                    type=int, default=2,
                                    help='Allow Event producer to run this many MVS ticks ahead before waiting.')
    human_mask_options.add_argument('--event-replay-mvs-pacing-max-wait-ms',
                                    dest='event_replay_mvs_pacing_max_wait_ms',
                                    type=float, default=250.0,
                                    help='Max wait per event slice when MVS replay is behind; prevents deadlock.')
    human_mask_options.add_argument('--event-replay-mvs-pacing-poll-ms',
                                    dest='event_replay_mvs_pacing_poll_ms',
                                    type=float, default=5.0,
                                    help='Polling interval while waiting for MVS replay target to catch up.')

    v26b_options = parser.add_argument_group('V26-B post-tracker polygon gate and business ID shadows')
    v26b_options.add_argument('--event-track-polygon-gate-mode', dest='event_track_polygon_gate_mode',
                              choices=['off', 'shadow', 'active'], default='off',
                              help='Post-tracker polygon gate. shadow preserves official LineFilter input.')
    v26b_options.add_argument('--event-track-polygon-gate-cameras', dest='event_track_polygon_gate_cameras',
                              type=str, default='', help='Comma-separated camera aliases; empty means all.')
    v26b_options.add_argument('--event-track-polygon-gate-hold-enable', dest='event_track_polygon_gate_hold_enable',
                              action='store_true', default=True)
    v26b_options.add_argument('--disable-event-track-polygon-gate-hold', dest='event_track_polygon_gate_hold_enable',
                              action='store_false')
    v26b_options.add_argument('--event-track-polygon-gate-hold-ms', dest='event_track_polygon_gate_hold_ms',
                              type=float, default=30.0)
    v26b_options.add_argument('--event-track-polygon-gate-hold-max-sync-gap',
                              dest='event_track_polygon_gate_hold_max_sync_gap', type=int, default=2)
    v26b_options.add_argument('--event-track-polygon-gate-dilate-px', dest='event_track_polygon_gate_dilate_px',
                              type=int, default=0)
    v26b_options.add_argument('--event-track-polygon-gate-log-jsonl', dest='event_track_polygon_gate_log_jsonl',
                              type=str, default='', help='Optional gate diagnostics JSONL; supports {camera}.')
    v26b_options.add_argument('--event-track-polygon-gate-log-max-fps', dest='event_track_polygon_gate_log_max_fps',
                              type=float, default=5.0)
    v26b_options.add_argument('--event-business-id-v1-shadow-enable', dest='event_business_id_v1_shadow_enable',
                              action='store_true', default=False)
    v26b_options.add_argument('--event-business-id-v1-shadow-cameras', dest='event_business_id_v1_shadow_cameras',
                              type=str, default='')
    v26b_options.add_argument('--event-business-id-v1-confirm-min-observations',
                              dest='event_business_id_v1_confirm_min_observations', type=int, default=4)
    v26b_options.add_argument('--event-business-id-v1-confirm-min-displacement-px',
                              dest='event_business_id_v1_confirm_min_displacement_px', type=float, default=20.0)
    v26b_options.add_argument('--event-business-id-v1-confirm-min-speed-px-ms',
                              dest='event_business_id_v1_confirm_min_speed_px_ms', type=float, default=0.32)
    v26b_options.add_argument('--event-business-id-v1-assoc-radius-px',
                              dest='event_business_id_v1_assoc_radius_px', type=float, default=28.0)
    v26b_options.add_argument('--event-business-id-v1-max-gap-ms',
                              dest='event_business_id_v1_max_gap_ms', type=float, default=120.0)
    v26b_options.add_argument('--event-business-id-v1-collision-window-ms',
                              dest='event_business_id_v1_collision_window_ms', type=float, default=260.0)
    v26b_options.add_argument('--event-business-id-v1-collision-max-gap-px',
                              dest='event_business_id_v1_collision_max_gap_px', type=float, default=95.0)
    v26b_options.add_argument('--event-business-id-v1-shadow-log-jsonl',
                              dest='event_business_id_v1_shadow_log_jsonl', type=str, default='')
    v26b_options.add_argument('--event-business-id-v1-shadow-log-max-fps',
                              dest='event_business_id_v1_shadow_log_max_fps', type=float, default=5.0)
    v26b_options.add_argument('--event-business-id-v3-mode', dest='event_business_id_v3_mode',
                              choices=['off', 'shadow'], default='off',
                              help='V3 behavior/size evaluator. Active authority is intentionally unavailable in V26-B.')
    v26b_options.add_argument('--event-business-id-v3-cameras', dest='event_business_id_v3_cameras', type=str, default='')
    v26b_options.add_argument('--event-business-id-v3-size-warmup-observations',
                              dest='event_business_id_v3_size_warmup_observations', type=int, default=6)
    v26b_options.add_argument('--event-business-id-v3-max-size-ratio',
                              dest='event_business_id_v3_max_size_ratio', type=float, default=3.0)
    v26b_options.add_argument('--event-business-id-v3-size-outlier-consecutive',
                              dest='event_business_id_v3_size_outlier_consecutive', type=int, default=3)
    v26b_options.add_argument('--event-business-id-v3-log-jsonl', dest='event_business_id_v3_log_jsonl',
                              type=str, default='')
    v26b_options.add_argument('--event-business-id-v3-log-max-fps', dest='event_business_id_v3_log_max_fps',
                              type=float, default=5.0)

    event_cpu_options = parser.add_argument_group('Event CPU / diagnostics optimization options')
    event_cpu_options.add_argument('--event-worker-status-enable', dest='event_worker_status_enable',
                                   action='store_true', default=True,
                                   help='写出 sync_ipc/event_worker_{camera}_status.json，供状态窗口和离线分析使用。')
    event_cpu_options.add_argument('--disable-event-worker-status', dest='event_worker_status_enable',
                                   action='store_false',
                                   help='关闭 event worker status JSON 写出。')
    event_cpu_options.add_argument('--event-worker-status-file', dest='event_worker_status_file', type=str,
                                   default='./sync_ipc/event_worker_{camera}_status.json')
    event_cpu_options.add_argument('--event-worker-status-interval-ms', dest='event_worker_status_interval_ms',
                                   type=float, default=1000.0)
    event_cpu_options.add_argument('--event-loop-profiler-enable', dest='event_loop_profiler_enable',
                                   action='store_true', default=False,
                                   help='Sandbox: write per-stage event hot-loop timing summaries.')
    event_cpu_options.add_argument('--disable-event-loop-profiler', dest='event_loop_profiler_enable',
                                   action='store_false')
    event_cpu_options.add_argument('--event-loop-profiler-interval-ms', dest='event_loop_profiler_interval_ms',
                                   type=float, default=1000.0)
    event_cpu_options.add_argument('--event-loop-profiler-jsonl-template', dest='event_loop_profiler_jsonl_template',
                                   type=str, default='./sync_ipc/event_loop_profile_{camera}.jsonl')
    event_cpu_options.add_argument('--event-worker-cpus', dest='event_worker_cpus', type=str, default='',
                                   help='按相机别名 A/B/C... 绑定 CPU，例如 30,31,32,33,34,35,36,37。')
    event_cpu_options.add_argument('--event-empty-slice-fastpath-enable', dest='event_empty_slice_fastpath_enable',
                                   action='store_true', default=True,
                                   help='空事件 slice 或人体 mask 过滤后为空时，跳过 spatter/line/render 等重处理。')
    event_cpu_options.add_argument('--disable-event-empty-slice-fastpath', dest='event_empty_slice_fastpath_enable',
                                   action='store_false')
    event_cpu_options.add_argument('--event-small-slice-event-thresh', dest='event_small_slice_event_thresh',
                                   type=int, default=0,
                                   help='预留参数：小事件 slice 阈值。0=仅空 slice 快速跳过。')
    event_cpu_options.add_argument('--event-overlay-pub-max-fps', dest='event_overlay_pub_max_fps', type=float,
                                   default=170.0, help='event overlay shm 发布最高 FPS；0=不限。')
    event_cpu_options.add_argument('--event-tsf1-pub-max-fps', dest='event_tsf1_pub_max_fps', type=float,
                                   default=170.0, help='TSF1 shared-memory 发布最高 FPS；0=不限。')
    event_cpu_options.add_argument('--event-obs-submit-max-fps', dest='event_obs_submit_max_fps', type=float,
                                   default=170.0, help='OBS/SRT 提交最高 FPS；0=不限。')
    event_cpu_options.add_argument('--event-viz-after-human-mask-filter', dest='event_viz_after_human_mask_filter',
                                   action='store_true', default=True,
                                   help='event overlay/OBS/TSF1 可视化数据源使用人体 mask 过滤后的事件。')
    event_cpu_options.add_argument('--disable-event-viz-after-human-mask-filter', dest='event_viz_after_human_mask_filter',
                                   action='store_false')
    event_cpu_options.add_argument('--event-viz-buffer-enable', dest='event_viz_buffer_enable', action='store_true', default=True,
                                   help='启用 50ms 可视化缓冲池，解耦事件主循环和 overlay/OBS/TSF1 发布。')
    event_cpu_options.add_argument('--disable-event-viz-buffer', dest='event_viz_buffer_enable', action='store_false')
    event_cpu_options.add_argument('--event-viz-buffer-ms', dest='event_viz_buffer_ms', type=float, default=50.0)
    event_cpu_options.add_argument('--event-viz-buffer-slots', dest='event_viz_buffer_slots', type=int, default=0,
                                   help='0=根据 buffer-ms / slice / output fps 自动计算。')
    event_cpu_options.add_argument('--event-viz-buffer-drop-policy', dest='event_viz_buffer_drop_policy', type=str, default='drop_oldest')
    event_cpu_options.add_argument('--event-viz-max-output-age-ms', dest='event_viz_max_output_age_ms', type=float, default=50.0)
    event_cpu_options.add_argument('--event-viz-build-max-fps', dest='event_viz_build_max_fps', type=float, default=0.0,
                                   help='Limit EventViz packet building before copying event arrays; 0=unlimited.')
    event_cpu_options.add_argument('--event-viz-skip-empty-fastpath', dest='event_viz_skip_empty_fastpath',
                                   action='store_true', default=False,
                                   help='Skip EventViz packet building for empty fastpath slices except idle heartbeat.')
    event_cpu_options.add_argument('--disable-event-viz-skip-empty-fastpath', dest='event_viz_skip_empty_fastpath',
                                   action='store_false')
    event_cpu_options.add_argument('--event-viz-worker-mode', dest='event_viz_worker_mode', type=str, default='thread', choices=['thread', 'off'])
    event_cpu_options.add_argument('--event-viz-no-mask-policy', dest='event_viz_no_mask_policy', type=str, default='empty', choices=['empty', 'filtered', 'raw_debug', 'skip'])
    event_cpu_options.add_argument('--event-viz-idle-heartbeat-fps', dest='event_viz_idle_heartbeat_fps', type=float, default=20.0)
    event_cpu_options.add_argument('--event-viz-sidecar-template', dest='event_viz_sidecar_template', type=str, default='./sync_ipc/event_viz_{camera}_latest.json')
    event_cpu_options.add_argument('--event-overlay-sidecar-template', dest='event_overlay_sidecar_template', type=str, default='./sync_ipc/event_overlay_{camera}_latest.json')
    event_cpu_options.add_argument(
        '--event-track-input-policy',
        dest='event_track_input_policy',
        type=str,
        default='post_mask',
        choices=['post_mask', 'raw', 'raw_on_mask_unavailable', 'raw_when_filtered_empty', 'raw_on_mask_unavailable_or_filtered_empty'],
        help='Input policy for STC/spatter/line_filter. post_mask preserves old behavior; raw decouples bullet tracking from human-mask visualization filtering.',
    )
    event_cpu_options.add_argument('--event-overlay-render-mode', dest='event_overlay_render_mode', type=str,
                                   default='bgr', choices=['bgr', 'mono_sparse'])
    event_cpu_options.add_argument('--event-overlay-mono-color', dest='event_overlay_mono_color', type=str,
                                   default='white', choices=['white', 'green', 'cyan'])
    event_cpu_options.add_argument('--event-overlay-mono-decay-us', dest='event_overlay_mono_decay_us', type=int, default=0)
    event_cpu_options.add_argument('--event-overlay-draw-debug-text', dest='event_overlay_draw_debug_text',
                                   action='store_true', default=True)
    event_cpu_options.add_argument('--disable-event-overlay-draw-debug-text', dest='event_overlay_draw_debug_text',
                                   action='store_false')
    event_cpu_options.add_argument('--event-obs-sidecar-enable', dest='event_obs_sidecar_enable', action='store_true', default=True)
    event_cpu_options.add_argument('--disable-event-obs-sidecar', dest='event_obs_sidecar_enable', action='store_false')
    event_cpu_options.add_argument('--event-obs-sidecar-template', dest='event_obs_sidecar_template', type=str, default='./sync_ipc/event_obs_{camera}_latest.json')
    event_cpu_options.add_argument('--event-overlay-shm-schema-version', dest='event_overlay_shm_schema_version', type=int, default=2)
    human_mask_options.add_argument('--event-human-mask-cache-enable', dest='event_human_mask_cache_enable',
                                    action='store_true', default=True,
                                    help='限制 human_result JSONL 轮询/解析频率，并复用已构造 mask。')
    human_mask_options.add_argument('--disable-event-human-mask-cache', dest='event_human_mask_cache_enable',
                                    action='store_false')
    human_mask_options.add_argument('--event-human-mask-cache-max-fps', dest='event_human_mask_cache_max_fps',
                                    type=float, default=40.0,
                                    help='human_result JSONL 轮询最高频率。默认 40Hz。')
    human_mask_options.add_argument('--event-human-mask-cache-stale-ms', dest='event_human_mask_cache_stale_ms',
                                    type=float, default=150.0,
                                    help='状态指标：human mask cache 超过该年龄认为 stale。')


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
        history_len=args.line_history_len,
        bootstrap_frames=args.line_bootstrap_frames,
        max_angle_deg=args.line_angle_thresh_deg,
        max_line_offset_px=args.line_max_offset_px,
        min_step_px=args.line_min_step_px,
        min_total_displacement_px=args.line_min_total_disp_px,
        max_missed_frames=args.line_max_missed_frames,
        same_direction_ratio_thresh=args.line_same_direction_ratio_thresh,
        bullet_min_output_streak=args.bullet_min_output_streak,
        min_valid_step_count=args.line_min_valid_step_count,
        min_valid_step_ratio=args.line_min_valid_step_ratio,
        # P0/P1
        maintain_max_offset_px=args.maintain_max_offset_px,
        maintain_max_static_frames=args.maintain_max_static_frames,
        maintain_max_bbox_std=args.maintain_max_bbox_std,
        probation_max_offset_px=args.probation_max_offset_px,
        ghost_capture_enabled=args.ghost_capture_enabled,
        # P2-A
        maintain_recent_offset_px=args.maintain_recent_offset_px,
        maintain_recent_offset_exceed_frames=args.maintain_recent_offset_exceed_frames,
        # P2-B
        maintain_bbox_ema_alpha=args.maintain_bbox_ema_alpha,
        maintain_bbox_ema_drift_ratio=args.maintain_bbox_ema_drift_ratio,
        # P2-C
        maintain_max_turn_count=args.maintain_max_turn_count,
        maintain_max_cumulative_turn_deg=args.maintain_max_cumulative_turn_deg,
        # P2-E / P3-1
        ghost_bounce_max_dist_px=args.ghost_bounce_max_dist_px,
        # P3-2 / P6-1
        ghost_bounce_min_speed_px_per_ms=args.ghost_bounce_min_speed_px_per_ms,
        same_track_bbox_reuse_min_disp_px=args.same_track_bbox_reuse_min_disp_px,
        # P3-3
        trigger_max_full_offset_ratio=args.trigger_max_full_offset_ratio,
        # P3-5
        trigger_max_sign_flip_ratio=args.trigger_max_sign_flip_ratio,
        # P5-1
        trigger_min_avg_speed_px_per_ms=args.trigger_min_avg_speed_px_per_ms,
        # P7-1
        trigger_raw_min_valid_steps=args.trigger_raw_min_valid_steps,
        trigger_sparse_burst_max_valid_steps=args.trigger_sparse_burst_max_valid_steps,
        trigger_sparse_burst_min_disp_per_step_px=args.trigger_sparse_burst_min_disp_per_step_px,
        trigger_sparse_burst_min_total_disp_px=args.trigger_sparse_burst_min_total_disp_px,
        trigger_sparse_burst_max_path_over_net=args.trigger_sparse_burst_max_path_over_net,
        trigger_old_raw_age_ms=args.trigger_old_raw_age_ms,
        trigger_old_raw_min_valid_steps=args.trigger_old_raw_min_valid_steps,
        # P7-2
        probation_new_min_extra_disp_px=args.probation_new_min_extra_disp_px,
        probation_new_min_extra_valid_steps=args.probation_new_min_extra_valid_steps,
        # P8
        model_track_enabled=args.model_track_enabled,
        model_soft_residual_px=args.model_soft_residual_px,
        model_hard_residual_px=args.model_hard_residual_px,
        model_outlier_residual_px=args.model_outlier_residual_px,
        model_correction_gain=args.model_correction_gain,
        model_weak_correction_gain=args.model_weak_correction_gain,
        model_max_accel_px_ms2=args.model_max_accel_px_ms2,
        model_outlier_kill_enabled=args.model_outlier_kill_enabled,
        model_outlier_kill_frames=args.model_outlier_kill_frames,
        # P13 raw_id 粘背景保护
        raw_id_sticky_guard_enabled=args.raw_id_sticky_guard_enabled,
        raw_id_sticky_guard_min_speed_px_ms=args.raw_id_sticky_guard_min_speed_px_ms,
        raw_id_sticky_guard_static_step_px=args.raw_id_sticky_guard_static_step_px,
        raw_id_sticky_guard_residual_px=args.raw_id_sticky_guard_residual_px,
        bullet_id_stitch_enabled=args.bullet_id_stitch_enabled,
        bullet_id_stitch_window_ms=args.bullet_id_stitch_window_ms,
        bullet_id_stitch_corridor_px=args.bullet_id_stitch_corridor_px,
        bullet_id_stitch_min_dir_cos=args.bullet_id_stitch_min_dir_cos,
        bullet_id_stitch_min_speed_px_ms=args.bullet_id_stitch_min_speed_px_ms,
        bullet_id_stitch_max_size_ratio=args.bullet_id_stitch_max_size_ratio,
        bullet_id_stitch_reverse_enabled=args.bullet_id_stitch_reverse_enabled,
        bullet_id_stitch_reverse_window_ms=args.bullet_id_stitch_reverse_window_ms,
        bullet_id_stitch_reverse_corridor_px=args.bullet_id_stitch_reverse_corridor_px,
        bullet_id_stitch_reverse_min_dir_cos=args.bullet_id_stitch_reverse_min_dir_cos,
        bullet_id_stitch_reverse_min_local_disp_px=args.bullet_id_stitch_reverse_min_local_disp_px,
        bullet_id_stitch_reverse_max_size_ratio=args.bullet_id_stitch_reverse_max_size_ratio,
        bullet_id_bounce_link_enabled=args.bullet_id_bounce_link_enabled,
        bullet_id_bounce_link_window_ms=args.bullet_id_bounce_link_window_ms,
        bullet_id_bounce_link_max_gap_px=args.bullet_id_bounce_link_max_gap_px,
        bullet_id_bounce_link_adaptive_max_px=args.bullet_id_bounce_link_adaptive_max_px,
        bullet_id_bounce_link_min_turn_angle_deg=args.bullet_id_bounce_link_min_turn_angle_deg,
        bullet_id_bounce_link_min_speed_px_ms=args.bullet_id_bounce_link_min_speed_px_ms,
        bullet_id_bounce_link_max_size_ratio=args.bullet_id_bounce_link_max_size_ratio,
        bullet_id_bounce_link_candidate_max_points=args.bullet_id_bounce_link_candidate_max_points,
    )
    assoc = AssociationConfig(
        max_distance_px=args.assoc_max_distance_px,
        direction_penalty_px=args.assoc_direction_penalty_px,
        max_size_ratio=args.assoc_max_size_ratio,
    )
    draw = DrawConfig(
        draw_trajectory=True,
        history_len=args.line_draw_history_len,
        pre_confirm_history_len=args.line_pre_confirm_history_len,
        hold_missed_frames=args.line_draw_hold_missed_frames,
        connect_max_gap_px=args.line_draw_connect_max_gap_px,
        kinematic_fit_enabled=args.line_draw_kinematic_fit_enabled,
        kinematic_force_straight_line=args.line_draw_kinematic_force_straight_line,
        kinematic_residual_px=args.line_draw_kinematic_residual_px,
        kinematic_max_curve_px=args.line_draw_kinematic_max_curve_px,
        kinematic_max_accel_px_per_ms2=args.line_draw_kinematic_max_accel_px_per_ms2,
        kinematic_sample_step_px=args.line_draw_kinematic_sample_step_px,
        draw_after_terminate_hold_ms=args.line_draw_after_terminate_hold_ms,
        extend_tail_after_terminate=args.line_draw_extend_tail_after_terminate,
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
        'digital_crop_requested': bool(
            args.enable_digital_crop or args.disable_digital_crop or args.crop_region is not None),
        'digital_crop_applied': False, 'digital_crop_enabled': None, 'digital_crop_region': None,
        'digital_crop_error': '',
        'afk_requested': bool(args.enable_afk or args.disable_afk),
        'afk_applied': False, 'afk_enabled': None, 'afk_error': '',
        'hw_trail_requested': bool(args.enable_hw_trail or args.disable_hw_trail),
        'hw_trail_applied': False, 'hw_trail_enabled': None, 'hw_trail_error': '',
        'erc_requested': bool(args.enable_erc or args.disable_erc or args.erc_cd_event_rate is not None),
        'erc_applied': False, 'erc_enabled': None, 'erc_error': '',
        'bias_print_requested': bool(args.print_biases), 'bias_print_done': False,
        'bias_updates_requested': list(args.bias_updates),
        'bias_updates_applied': [], 'bias_updates_failed': [], 'bias_error': '',
        'native_hal_bridge_requested': bool(getattr(args, 'event_native_hal_bridge_enable', False)),
        'native_hal_bridge_active': False,
        'native_hal_bridge_bin': '',
        'native_hal_bridge_stderr_log': '',
        'native_hal_bridge_note': '',
        'native_hal_fifo_requested': bool(str(getattr(args, 'event_native_hal_fifo_path', '') or '').strip()),
        'native_hal_fifo_active': False,
        'native_hal_fifo_path': '',
        'native_hal_fifo_note': '',
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
        ok_en = crop.enable(True)
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
    fifo_path_arg = str(getattr(args, 'event_native_hal_fifo_path', '') or '').strip()
    if live and fifo_path_arg:
        if HalEventFifoIterator is None:
            raise RuntimeError(
                'V6-C native HAL FIFO requested, but native_event_bridge.hal_event_pipe_iterator import failed.'
            )
        camera_alias = _get_camera_short_alias(args)
        serial = str(getattr(args, 'tsf1_camera_id', '') or '').strip()
        try:
            fifo_path = fifo_path_arg.format(camera=camera_alias or serial or 'cam', serial=serial)
        except Exception:
            fifo_path = fifo_path_arg
        fifo_path = _build_alias_output_path(fifo_path, camera_alias, serial, default_ext='.evb')
        hal_status['native_hal_fifo_path'] = fifo_path
        if not os.path.exists(fifo_path):
            raise FileNotFoundError(f'V6-C native HAL FIFO not found: {fifo_path}')

        unsupported = []
        if bool(getattr(args, 'enable_digital_crop', False)) or bool(getattr(args, 'disable_digital_crop', False)):
            unsupported.append('digital_crop')
        if bool(getattr(args, 'enable_afk', False)) or bool(getattr(args, 'disable_afk', False)):
            unsupported.append('afk')
        if bool(getattr(args, 'enable_erc', False)) or bool(getattr(args, 'disable_erc', False)):
            unsupported.append('erc_applied_by_broker')
        if bool(getattr(args, 'enable_hw_trail', False)) or bool(getattr(args, 'disable_hw_trail', False)):
            unsupported.append('trail_applied_by_broker')
        if bool(getattr(args, 'enable_ext_trigger_in', False)):
            unsupported.append('trigger_in_applied_by_broker')
        if unsupported:
            hal_status['native_hal_fifo_note'] = 'worker_hal_tuning_bypassed=' + ','.join(unsupported)

        mv_iterator = HalEventFifoIterator(fifo_path=fifo_path)
        hal_status['native_hal_fifo_active'] = True
        hal_status['digital_crop_error'] = hal_status['digital_crop_error'] or 'not applied in native FIFO worker'
        hal_status['afk_error'] = hal_status['afk_error'] or 'not applied in native FIFO worker'
        print(
            f'[HALFifo] V6-C native FIFO active serial={serial} fifo={fifo_path}',
            flush=True,
        )
        return mv_iterator, True, None
    if live and bool(getattr(args, 'event_native_hal_bridge_enable', False)):
        if HalEventPipeIterator is None:
            raise RuntimeError(
                'V6-B native HAL bridge requested, but native_event_bridge.hal_event_pipe_iterator import failed.'
            )
        bridge_bin = str(getattr(args, 'event_native_hal_bridge_bin', '') or '').strip()
        if not bridge_bin:
            bridge_bin = str((Path(__file__).resolve().parent / 'native_event_bridge' / 'build' /
                              'hal_event_pipe_bridge').resolve())
        if not os.path.exists(bridge_bin):
            raise FileNotFoundError(
                f'V6-B native HAL bridge binary not found: {bridge_bin}. '
                f'Build it with: {Path(__file__).resolve().parent / "native_event_bridge" / "build_native_event_bridge.sh"}'
            )
        camera_alias = _get_camera_short_alias(args)
        serial = str(getattr(args, 'tsf1_camera_id', '') or '').strip()
        stderr_tmpl = str(getattr(args, 'event_native_hal_bridge_stderr_log', '') or '').strip()
        stderr_log = ''
        if stderr_tmpl:
            try:
                stderr_log = stderr_tmpl.format(camera=camera_alias or serial or 'cam', serial=serial)
            except Exception:
                stderr_log = stderr_tmpl
            stderr_log = _build_alias_output_path(stderr_log, camera_alias, serial, default_ext='.log')

        hal_status['native_hal_bridge_bin'] = bridge_bin
        hal_status['native_hal_bridge_stderr_log'] = stderr_log
        unsupported = []
        if bool(getattr(args, 'enable_digital_crop', False)) or bool(getattr(args, 'disable_digital_crop', False)):
            unsupported.append('digital_crop')
        if bool(getattr(args, 'enable_afk', False)) or bool(getattr(args, 'disable_afk', False)):
            unsupported.append('afk')
        if unsupported:
            hal_status['native_hal_bridge_note'] = 'unsupported_hal_tuning=' + ','.join(unsupported)
            print(f'[HALBridge][WARN] V6-B bridge does not apply: {",".join(unsupported)}', flush=True)

        mv_iterator = HalEventPipeIterator(
            bridge_bin=bridge_bin,
            serial=serial,
            slice_us=int(getattr(args, 'accumulation_time_us', 4000) or 4000),
            wait_mode=str(getattr(args, 'event_native_hal_bridge_wait_mode', 'poll') or 'poll'),
            poll_sleep_us=int(getattr(args, 'event_native_hal_bridge_poll_sleep_us', 100) or 0),
            enable_trigger_in=bool(getattr(args, 'enable_ext_trigger_in', False)),
            enable_erc=bool(getattr(args, 'enable_erc', False)) and not bool(getattr(args, 'disable_erc', False)),
            erc_rate=getattr(args, 'erc_cd_event_rate', None),
            enable_trail=bool(getattr(args, 'enable_hw_trail', False)) and not bool(getattr(args, 'disable_hw_trail', False)),
            trail_type=str(getattr(args, 'hw_trail_type', 'stc_keep_trail') or 'stc_keep_trail'),
            trail_threshold_us=int(getattr(args, 'hw_trail_threshold_us', 10000) or 10000),
            biases=list(getattr(args, 'bias_updates', []) or []),
            stderr_log=stderr_log,
        )
        hal_status['native_hal_bridge_active'] = True
        hal_status['digital_crop_error'] = hal_status['digital_crop_error'] or 'not applied in native bridge'
        hal_status['afk_error'] = hal_status['afk_error'] or 'not applied in native bridge'
        print(
            f'[HALBridge] V6-B native bridge active serial={serial} bin={bridge_bin} '
            f'wait_mode={getattr(args, "event_native_hal_bridge_wait_mode", "poll")}',
            flush=True,
        )
        return mv_iterator, True, None
    if live:
        # 多相机支持：当指定了 tsf1_camera_id（序列号）且未指定 event_file_path 时，
        # 将序列号直接传给 initiate_device，使其按序列号打开指定相机。
        # initiate_device 内部逻辑：path 非空且不是文件时，会将其作为序列号传给
        # DeviceDiscovery.open(path)。
        # 多进程同时打开相机时 HAL 可能存在竞争，因此加入重试机制。
        camera_serial = getattr(args, 'tsf1_camera_id', '') or ''
        open_path = args.event_file_path  # 默认空字符串 → 打开第一台
        if not args.event_file_path and camera_serial and camera_serial != 'cam0':
            open_path = camera_serial  # 传序列号让 initiate_device 按序列号打开

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
        self.w = int(width)
        self.h = int(height)
        self.writer = writer
        self.args = args
        self.interval_us = int(round(1_000_000.0 / max(0.1, float(args.tsf1_pub_fps))))
        self.last_ts = np.full((self.h, self.w), -1, dtype=np.int64)
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
        img = np.zeros((self.h, self.w), dtype=np.uint8)
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
        beta = float(getattr(self.args, 'tsf1_post_beta', 0.0))
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

    def poll(self, max_records=None):
        out = []
        limit = None
        try:
            if max_records is not None:
                limit = max(1, int(max_records))
        except Exception:
            limit = None
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
                        if limit is not None and len(out) >= limit:
                            break
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
        self.camera_alias = str(
            camera_alias or getattr(args, 'hit_camera_alias', '') or getattr(args, 'tsf1_camera_id', '') or '')
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
        try:
            self.stats_jsonl_max_fps = max(0.0, float(getattr(args, 'event_human_mask_stats_max_fps', 2.0) or 0.0))
        except Exception:
            self.stats_jsonl_max_fps = 2.0
        try:
            self.stats_jsonl_flush_ms = max(0.0, float(getattr(args, 'event_human_mask_stats_flush_ms', 1000.0) or 0.0))
        except Exception:
            self.stats_jsonl_flush_ms = 1000.0
        self._next_stats_jsonl_mono = 0.0
        self._last_stats_jsonl_flush_mono = 0.0
        self._stats_jsonl_write_count = 0
        self._stats_jsonl_skip_count = 0
        if stats_tmpl and stats_tmpl.lower() not in ('0', 'false', 'none', 'no', 'off'):
            try:
                self.stats_jsonl_path = stats_tmpl.format(camera=self.camera_alias, cam=self.camera_alias,
                                                          id=self.camera_alias, alias=self.camera_alias)
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
            self.jsonl_path = tmpl.format(camera=self.camera_alias, cam=self.camera_alias, id=self.camera_alias,
                                          alias=self.camera_alias)
        except Exception:
            self.jsonl_path = tmpl
        self.tailer = _JsonlTailerLite(self.jsonl_path) if self.enabled and self.jsonl_path else None
        self.cache_enable = bool(getattr(args, 'event_human_mask_cache_enable', True))
        self.cache_max_fps = max(0.1, float(getattr(args, 'event_human_mask_cache_max_fps', 40.0) or 40.0))
        self.cache_stale_ms = max(0.0, float(getattr(args, 'event_human_mask_cache_stale_ms', 150.0) or 150.0))
        self._next_jsonl_poll_mono = 0.0
        self._jsonl_poll_count = 0
        self._jsonl_poll_skip_count = 0
        self._jsonl_records_loaded = 0
        self._last_jsonl_poll_wall_us = 0

        config_path = str(
            getattr(args, 'event_human_mask_mvs_config', 'hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json') or '')
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

    def _poll(self, force: bool = False, max_records=None):
        if not self.enabled or self.tailer is None:
            return 0
        if self.cache_enable and not force:
            now_mono = time.monotonic()
            if now_mono < self._next_jsonl_poll_mono:
                self._jsonl_poll_skip_count += 1
                return 0
            self._next_jsonl_poll_mono = now_mono + 1.0 / self.cache_max_fps
        self._jsonl_poll_count += 1
        self._last_jsonl_poll_wall_us = int(time.time() * 1_000_000)
        n = 0
        for rec in self.tailer.poll(max_records=max_records):
            sync_i = self._record_sync_index(rec)
            if sync_i is None:
                continue
            self._records[int(sync_i)] = rec
            self._latest_sync = int(sync_i)
            self._latest_record = rec
            n += 1
        self._jsonl_records_loaded += int(n)
        if len(self._records) > self.max_records:
            keys = sorted(self._records.keys())
            for k in keys[:max(0, len(keys) - self.max_records)]:
                self._records.pop(k, None)
                self._mask_cache.pop(k, None)
        return n

    def poll_records(self, force: bool = False):
        return self._poll(force=force)

    def poll_records_until_sync(self, sync_index, force: bool = False, chunk_records: int = 128, max_chunks: int = 64):
        if not self.enabled or sync_index is None:
            return 0
        try:
            target = int(sync_index)
        except Exception:
            return 0
        if target in self._records:
            return 0
        loaded = 0
        chunks = max(1, int(max_chunks or 1))
        chunk_size = max(1, int(chunk_records or 1))
        for _ in range(chunks):
            n = int(self._poll(force=force, max_records=chunk_size) or 0)
            loaded += n
            if target in self._records:
                break
            if n <= 0:
                break
            try:
                if self._latest_sync is not None and int(self._latest_sync) >= target:
                    break
            except Exception:
                break
        return loaded

    def has_record_for_sync(self, sync_index):
        if not self.enabled or sync_index is None:
            return False
        try:
            sync_i = int(sync_index)
        except Exception:
            return False
        rec = self._records.get(sync_i)
        if rec is None:
            self.poll_records_until_sync(sync_i, force=False)
            rec = self._records.get(sync_i)
        if rec is None:
            return False
        ok_rec, _ = self._record_allowed_for_hard_filter(rec)
        return bool(ok_rec)

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
            return False, f'current_frame_expired:dt_ms={dt_us / 1000.0:.1f}'
        return True, f'frame_window:dt_ms={dt_us / 1000.0:.1f}'

    def _get_record_for_sync(self, sync_index, sync_event_ts_us=None, slice_ts_us=0):
        """
        Get the human record for the current MVS sync_index.

        Important: this is not a one-shot match. 4ms event slices inside the same
        40ms MVS frame reuse the same current-frame mask. However, a mask from
        sync_index=N is never used after the event stream has advanced to N+1.
        """
        if sync_index is None:
            self._poll()
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

        self.poll_records_until_sync(sync_i, force=False)

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
                self.poll_records_until_sync(sync_i, force=True)
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
            emitted = False
            for poly in polys:
                if isinstance(poly, list) and len(poly) >= 3:
                    emitted = True
                    yield poly
            if emitted:
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
        invalid_polygon_count = 0
        bbox_fallback_count = 0
        used_bbox_as_polygon = 0
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
            polys_raw = person.get('mask_polygons') or person.get('polygons') or person.get('poly') or []
            if isinstance(polys_raw, dict):
                polys_raw = list(polys_raw.values())
            valid_poly_count = 0
            if isinstance(polys_raw, list):
                for poly_raw in polys_raw:
                    if isinstance(poly_raw, list) and len(poly_raw) >= 3:
                        valid_poly_count += 1
                    else:
                        invalid_polygon_count += 1
            box = person.get('bbox_xyxy') or person.get('bbox') or None
            bbox_can_fallback = isinstance(box, (list, tuple)) and len(box) >= 4
            if isinstance(polys_raw, list) and polys_raw and valid_poly_count <= 0 and bbox_can_fallback:
                bbox_fallback_count += 1
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
                    if valid_poly_count <= 0 and bbox_can_fallback:
                        used_bbox_as_polygon += 1
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
        def _rec_int(*keys, default=-1):
            for key in keys:
                try:
                    v = rec.get(key)
                    if v is not None:
                        return int(float(v))
                except Exception:
                    pass
            return int(default)
        def _rec_wall_us(*keys):
            mm = rec.get('mvs_meta') if isinstance(rec.get('mvs_meta'), dict) else {}
            for key in keys:
                for src in (rec, mm):
                    try:
                        v = src.get(key)
                        if v is not None:
                            return int(float(v))
                    except Exception:
                        pass
            return 0
        info = {
            'sync_index': int(sync_i) if sync_i is not None else None,
            'infer_source': self._record_infer_source(rec),
            'human_carry_age_sync': rec.get('human_carry_age_sync') if isinstance(rec, dict) else None,
            'person_count': int(len(persons)),
            'used_persons': int(used_persons),
            'used_polygons': int(used_polys),
            'invalid_polygon_count': int(invalid_polygon_count),
            'bbox_fallback_count': int(bbox_fallback_count),
            'used_bbox_as_polygon': int(used_bbox_as_polygon),
            'mask_pixels': int(np.count_nonzero(mask)),
            'human_frame_id': _rec_int('frame_id_local', 'mvs_frame_num', 'sync_index_hint', default=(sync_i if sync_i is not None else -1)),
            'human_mvs_frame_id': _rec_int('mvs_frame_num', 'frame_id_local', 'sync_index_hint', default=(sync_i if sync_i is not None else -1)),
            'human_capture_wall_us': _rec_wall_us('capture_wall_us', 'timestamp_us', 'host_read_wall_us'),
            'human_record_wall_us': _rec_wall_us('record_wall_us', 'publish_wall_us', 'receiver_wall_us'),
        }
        if sync_i is not None:
            self._mask_cache[int(sync_i)] = (mask, info)
        return mask, info

    def _update_stats(self, slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_out, n_drop, info,
                      reason, applied, extra=None):
        drop_ratio = float(n_drop) / float(max(1, n_in))
        stats = {
            'enabled': bool(self.enabled),
            'mode': str(self.mode),
            'applied': bool(applied),
            'reason': str(reason),
            'camera': self.camera_alias,
            'slice_ts_us': int(slice_ts_us or 0),
            'current_sync_index': None if current_sync_index is None else int(current_sync_index),
            'current_mvs_sync_index': None if current_sync_index is None else int(current_sync_index),
            'current_sync_event_ts_us': None if current_sync_event_ts_us is None else int(current_sync_event_ts_us),
            'latest_record_sync': None if self._latest_sync is None else int(self._latest_sync),
            'current_to_latest_record_sync_delta': None,
            'match_ok': bool(isinstance(info, dict)),
            'mask_available': bool(isinstance(info, dict) and int(info.get('mask_pixels', 0) or 0) > 0),
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
            'cache_enable': bool(self.cache_enable),
            'cache_max_fps': float(self.cache_max_fps),
            'cache_stale_ms': float(self.cache_stale_ms),
            'jsonl_poll_count': int(self._jsonl_poll_count),
            'jsonl_poll_skip_count': int(self._jsonl_poll_skip_count),
            'jsonl_records_loaded': int(self._jsonl_records_loaded),
            'last_jsonl_poll_wall_us': int(self._last_jsonl_poll_wall_us),
            'stats_jsonl_enabled': self._stats_fp is not None,
            'stats_jsonl_path': str(self.stats_jsonl_path or ''),
            'stats_jsonl_max_fps': float(getattr(self, 'stats_jsonl_max_fps', 0.0)),
            'stats_jsonl_write_count': int(getattr(self, '_stats_jsonl_write_count', 0)),
            'stats_jsonl_skip_count': int(getattr(self, '_stats_jsonl_skip_count', 0)),
        }
        try:
            if current_sync_index is not None and self._latest_sync is not None:
                stats['current_to_latest_record_sync_delta'] = int(self._latest_sync) - int(current_sync_index)
        except Exception:
            stats['current_to_latest_record_sync_delta'] = None
        if isinstance(info, dict):
            for key in ('sync_index', 'infer_source', 'human_carry_age_sync', 'person_count', 'used_persons',
                        'used_polygons', 'invalid_polygon_count', 'bbox_fallback_count', 'used_bbox_as_polygon',
                        'mask_pixels', 'human_frame_id', 'human_mvs_frame_id',
                        'human_capture_wall_us', 'human_record_wall_us'):
                if key in info:
                    stats[key] = info.get(key)
        if isinstance(extra, dict):
            for key, value in extra.items():
                stats[str(key)] = value
        self._last_stats = stats
        if self._stats_fp is not None:
            try:
                now_mono = time.monotonic()
                max_fps = float(getattr(self, 'stats_jsonl_max_fps', 0.0) or 0.0)
                if max_fps > 0.0 and now_mono < self._next_stats_jsonl_mono:
                    self._stats_jsonl_skip_count += 1
                    return
                if max_fps > 0.0:
                    self._next_stats_jsonl_mono = now_mono + 1.0 / max_fps
                self._stats_jsonl_write_count += 1
                stats['stats_jsonl_write_count'] = int(self._stats_jsonl_write_count)
                stats['stats_jsonl_skip_count'] = int(self._stats_jsonl_skip_count)
                self._stats_fp.write(json.dumps(stats, ensure_ascii=False, separators=(',', ':')) + '\n')
                flush_s = float(getattr(self, 'stats_jsonl_flush_ms', 1000.0) or 0.0) / 1000.0
                if flush_s <= 0.0 or now_mono - self._last_stats_jsonl_flush_mono >= flush_s:
                    self._stats_fp.flush()
                    self._last_stats_jsonl_flush_mono = now_mono
            except Exception:
                pass

    def get_last_stats(self):
        return dict(getattr(self, '_last_stats', {}) or {})

    def get_current_mask(self, current_sync_index, slice_ts_us, current_sync_event_ts_us=None):
        """Return the strict-current-frame polygon mask without filtering events.

        V26-B uses this as a provider for post-tracker cluster gating. Keeping
        this read path separate from ``filter_events`` ensures the raw event
        tracking stream is unchanged while the gate is in shadow mode.
        """
        if not self.enabled:
            return None, None, 'provider_disabled'
        rec, reason = self._get_record_for_sync(
            current_sync_index,
            current_sync_event_ts_us,
            slice_ts_us,
        )
        if rec is None:
            self._last_reason = str(reason)
            return None, None, str(reason)
        mask, info = self._build_mask_for_record(rec)
        if isinstance(info, dict) and info.get('sync_index') is not None:
            self._active_mask_info = dict(info)
        if mask is None or not isinstance(info, dict) or int(info.get('mask_pixels', 0) or 0) <= 0:
            self._last_reason = 'empty_current_mask'
            return None, info, self._last_reason
        self._last_reason = str(reason)
        return mask, info, str(reason)

    def filter_events(self, evs, current_sync_index, slice_ts_us, current_sync_event_ts_us=None, stats_extra=None):
        mode_s = str(getattr(self, 'mode', 'hard_events') or 'hard_events').strip().lower()
        if mode_s in ('off', 'none', 'disabled', 'disable'):
            try:
                n0 = int(len(evs)) if evs is not None else 0
            except Exception:
                n0 = 0
            extra = dict(stats_extra or {}) if isinstance(stats_extra, dict) else {}
            extra['mode_effective'] = 'off'
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n0, n0, 0, None, 'mode_off',
                               False, extra)
            return evs
        if (not self.enabled) or evs is None:
            try:
                n0 = int(len(evs)) if evs is not None else 0
            except Exception:
                n0 = 0
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n0, n0, 0, None, 'disabled',
                               False, stats_extra)
            return evs
        n_in = int(len(evs))
        if n_in <= 0:
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, None,
                               'empty_events', False, stats_extra)
            return evs
        rec, reason = self._get_record_for_sync(current_sync_index, current_sync_event_ts_us, slice_ts_us)
        if rec is None:
            self._last_reason = reason
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, None, reason,
                               False, stats_extra)
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
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info,
                               self._last_reason, False, stats_extra)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, self._last_reason)
            return evs
        try:
            xs = evs['x'].astype(np.int32, copy=False)
            ys = evs['y'].astype(np.int32, copy=False)
        except Exception:
            self._last_reason = 'event_dtype_missing_xy'
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info,
                               self._last_reason, False, stats_extra)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, self._last_reason)
            return evs
        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        inside = np.zeros(n_in, dtype=bool)
        inside[valid] = mask[ys[valid], xs[valid]] > 0
        mode_visual_only = mode_s in ('visual_only', 'visual-only', 'stats_only', 'stats-only', 'observe', 'soft')
        if mode_visual_only:
            would_drop = int(np.count_nonzero(inside))
            extra = dict(stats_extra or {}) if isinstance(stats_extra, dict) else {}
            extra['mode_effective'] = 'visual_only'
            extra['would_drop_count'] = int(would_drop)
            extra['would_drop_pct'] = round((float(would_drop) / float(max(1, n_in))) * 100.0, 3)
            self._last_reason = f'{reason}:visual_only'
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info,
                               self._last_reason, False, extra)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, self._last_reason)
            return evs
        keep = ~inside
        n_drop = int(np.count_nonzero(inside))
        if n_drop <= 0:
            self._last_reason = reason
            self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_in, 0, info, reason,
                               False, stats_extra)
            self._maybe_print(slice_ts_us, n_in, n_in, 0, info, reason)
            return evs
        out = evs[keep]
        n_out = int(len(out))
        self._total_in += n_in
        self._total_out += n_out
        self._total_drop += n_drop
        self._last_reason = reason
        self._update_stats(slice_ts_us, current_sync_index, current_sync_event_ts_us, n_in, n_out, n_drop, info, reason,
                           True, stats_extra)
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
class HumanMaskSyncEventBuffer:
    """Delay event slices by sync_index so the same-sync human mask can catch up."""

    def __init__(self, args, camera_alias: str):
        mask_enabled = bool(getattr(args, 'event_human_mask_filter_enable', False))
        self.enabled = bool(mask_enabled and getattr(args, 'event_human_mask_buffer_enable', False))
        self.camera_alias = str(camera_alias or '')
        self.buffer_ms = max(0.0, float(getattr(args, 'event_human_mask_buffer_ms', 90.0) or 0.0))
        self.timeout_ms = max(self.buffer_ms, float(getattr(args, 'event_human_mask_buffer_timeout_ms', 150.0) or 0.0))
        self.filter_max_fps = max(0.0, float(getattr(args, 'event_human_mask_filter_max_fps', 0.0) or 0.0))
        if self.filter_max_fps > 0.0:
            self.buffer_ms = max(self.buffer_ms, 1000.0 / self.filter_max_fps)
            self.timeout_ms = max(self.timeout_ms, self.buffer_ms)
        self.buffer_us = int(round(self.buffer_ms * 1000.0))
        self.timeout_us = int(round(self.timeout_ms * 1000.0))
        self.release_policy = str(
            getattr(args, 'event_human_mask_buffer_release_policy', 'raw_at_budget') or 'raw_at_budget'
        ).lower()
        if self.release_policy not in ('raw_at_budget', 'raw_at_timeout', 'strict_same_sync_drop'):
            self.release_policy = 'raw_at_budget'
        self.strict_same_sync = self.release_policy == 'strict_same_sync_drop'
        self.max_events = max(1, int(getattr(args, 'event_human_mask_buffer_max_events', 300000) or 300000))
        self.max_buckets = max(1, int(getattr(args, 'event_human_mask_buffer_max_buckets', 8) or 8))
        self.empty_release_fps = max(0.0, float(getattr(args, 'event_human_mask_buffer_empty_release_fps', 0.0) or 0.0))
        self._empty_release_limiter = _SimpleRateLimiter(self.empty_release_fps, f'human_mask_empty_{self.camera_alias}')
        self._empty_held_slices = 0
        self._empty_skipped_slices = 0
        self._buckets = collections.OrderedDict()
        self._ready = collections.deque()
        self._depth_events = 0
        self._submitted_slices = 0
        self._released_slices = 0
        self._forced_releases = 0
        self._dropped_slices = 0
        self._dropped_events = 0
        self._max_depth_events = 0
        self._max_depth_buckets = 0
        self._release_counts = collections.Counter()
        self._drop_counts = collections.Counter()
        self._last_stats = {'enabled': bool(self.enabled), 'reason': 'init'}
        if self.enabled:
            print(
                f"[HumanMaskBuffer][{self.camera_alias}] enabled buffer_ms={self.buffer_ms:.1f} "
                f"timeout_ms={self.timeout_ms:.1f} policy={self.release_policy} "
                f"max_events={self.max_events} max_buckets={self.max_buckets} "
                f"empty_release_fps={self.empty_release_fps:.1f}",
                flush=True,
            )

    @staticmethod
    def _event_count(evs):
        try:
            return int(len(evs)) if evs is not None else 0
        except Exception:
            return 0

    @staticmethod
    def _copy_events(evs):
        if evs is None:
            return evs
        try:
            return evs.copy()
        except Exception:
            return evs

    @staticmethod
    def _sync_int(sync_index):
        if sync_index is None:
            return None
        try:
            return int(sync_index)
        except Exception:
            return None

    def submit(self, raw_evs, *, sync_index, sync_event_ts_us, slice_ts_us, wall_time_us, sync_diag=None):
        if not self.enabled:
            return
        self._submitted_slices += 1
        raw_n = self._event_count(raw_evs)
        sync_i = self._sync_int(sync_index)
        item = {
            'evs': raw_evs if raw_n <= 0 else self._copy_events(raw_evs),
            'raw_n': int(raw_n),
            'sync_index': sync_i,
            'sync_event_ts_us': sync_event_ts_us,
            'slice_ts_us': int(slice_ts_us or 0),
            'wall_time_us': int(wall_time_us or 0),
            'sync_diag': dict(sync_diag or {}) if isinstance(sync_diag, dict) else {},
        }
        if raw_n <= 0:
            if self.empty_release_fps > 0.0:
                self._empty_held_slices += 1
                if not self._empty_release_limiter.allow():
                    self._empty_skipped_slices += 1
                    self._refresh_stats('empty_held')
                    return
                item = dict(item)
                item['merged_slices'] = int(max(1, self._empty_held_slices))
                self._empty_held_slices = 0
                reason = 'empty_heartbeat'
            else:
                reason = 'empty_immediate'
            self._ready.append((item, reason, False))
            self._refresh_stats(reason)
            return
        if sync_i is None:
            if self.strict_same_sync:
                self._dropped_slices += 1
                self._dropped_events += int(raw_n)
                self._drop_counts['sync_unlocked_drop'] += 1
                self._refresh_stats('sync_unlocked_drop')
                return
            reason = 'no_sync_immediate'
            self._ready.append((item, reason, False))
            self._refresh_stats(reason)
            return
        bucket = self._buckets.get(sync_i)
        if bucket is None:
            bucket = {'sync_index': sync_i, 'items': collections.deque(), 'event_count': 0}
            self._buckets[sync_i] = bucket
        bucket['items'].append(item)
        bucket['event_count'] += int(raw_n)
        self._depth_events += int(raw_n)
        self._max_depth_events = max(self._max_depth_events, self._depth_events)
        self._max_depth_buckets = max(self._max_depth_buckets, len(self._buckets))
        self._refresh_stats('submitted')

    def _oldest_bucket(self):
        while self._buckets:
            sync_i = next(iter(self._buckets))
            bucket = self._buckets.get(sync_i)
            if bucket is None or not bucket.get('items'):
                self._buckets.pop(sync_i, None)
                continue
            return sync_i, bucket
        return None, None

    def _release_item(self, item, release_reason, forced, mask_filter, now_wall_us):
        age_us = max(0, int(now_wall_us or 0) - int(item.get('wall_time_us') or 0))
        extra = {
            'buffer_enabled': bool(self.enabled),
            'buffer_wait_ms': float(self.buffer_ms),
            'buffer_timeout_ms': float(self.timeout_ms),
            'buffer_release_policy': str(self.release_policy),
            'buffer_release_reason': str(release_reason),
            'buffer_forced_release': bool(forced),
            'buffer_slice_age_ms': round(age_us / 1000.0, 3),
            'buffer_bucket_sync_index': item.get('sync_index'),
            'buffer_depth_events': int(self._depth_events),
            'buffer_depth_buckets': int(len(self._buckets)),
            'buffer_submitted_slices': int(self._submitted_slices),
            'buffer_released_slices': int(self._released_slices + 1),
            'mask_gate_mode': 'strict_same_sync' if self.strict_same_sync else 'fallback_raw',
            'raw_fallback_enabled': not bool(self.strict_same_sync),
        }
        if isinstance(item.get('sync_diag'), dict):
            extra.update(item.get('sync_diag') or {})
        detection_evs = mask_filter.filter_events(
            item.get('evs'),
            current_sync_index=item.get('sync_index'),
            current_sync_event_ts_us=item.get('sync_event_ts_us'),
            slice_ts_us=int(item.get('slice_ts_us') or 0),
            stats_extra=extra,
        )
        detection_n = self._event_count(detection_evs)
        self._released_slices += 1
        if forced:
            self._forced_releases += 1
        self._release_counts[str(release_reason)] += 1
        self._refresh_stats(str(release_reason))
        return {
            'raw_evs': item.get('evs'),
            'detection_evs': detection_evs,
            'raw_n': int(item.get('raw_n') or 0),
            'detection_n': int(detection_n),
            'ts': int(item.get('slice_ts_us') or 0),
            'capture_wall_time_us': int(now_wall_us or 0),
            'sync_index': item.get('sync_index'),
            'sync_event_ts_us': item.get('sync_event_ts_us'),
            'release_reason': str(release_reason),
            'forced_release': bool(forced),
        }

    def _release_oldest_bucket_item(self, bucket, release_reason, forced, mask_filter, now_wall_us):
        item = bucket['items'].popleft()
        raw_n = int(item.get('raw_n') or 0)
        bucket['event_count'] = max(0, int(bucket.get('event_count') or 0) - raw_n)
        self._depth_events = max(0, int(self._depth_events) - raw_n)
        sync_i = bucket.get('sync_index')
        if not bucket['items']:
            self._buckets.pop(sync_i, None)
        return self._release_item(item, release_reason, forced, mask_filter, now_wall_us)

    def _release_bucket_items(self, bucket, release_reason, forced, mask_filter, now_wall_us):
        items = list(bucket.get('items') or [])
        sync_i = bucket.get('sync_index')
        self._buckets.pop(sync_i, None)
        if not items:
            self._refresh_stats(str(release_reason))
            return None
        raw_parts = [it.get('evs') for it in items if self._event_count(it.get('evs')) > 0]
        try:
            if len(raw_parts) == 0:
                raw_evs = items[-1].get('evs')
            elif len(raw_parts) == 1:
                raw_evs = raw_parts[0]
            else:
                raw_evs = np.concatenate(raw_parts)
        except Exception:
            raw_evs = raw_parts[0] if raw_parts else items[-1].get('evs')
        raw_n = self._event_count(raw_evs)
        self._depth_events = max(0, int(self._depth_events) - sum(int(it.get('raw_n') or 0) for it in items))
        last_item = items[-1]
        first_item = items[0]
        merged = {
            'evs': raw_evs,
            'raw_n': int(raw_n),
            'sync_index': sync_i,
            'sync_event_ts_us': last_item.get('sync_event_ts_us'),
            'slice_ts_us': int(last_item.get('slice_ts_us') or 0),
            'wall_time_us': int(first_item.get('wall_time_us') or last_item.get('wall_time_us') or 0),
            'merged_slices': int(len(items)),
            'sync_diag': dict(last_item.get('sync_diag') or {}) if isinstance(last_item.get('sync_diag'), dict) else {},
        }
        released = self._release_item(merged, release_reason, forced, mask_filter, now_wall_us)
        if isinstance(released, dict):
            released['merged_slices'] = int(len(items))
        return released

    def _drop_bucket_items(self, bucket, drop_reason):
        items = list(bucket.get('items') or [])
        sync_i = bucket.get('sync_index')
        self._buckets.pop(sync_i, None)
        dropped_events = sum(int(it.get('raw_n') or 0) for it in items)
        self._depth_events = max(0, int(self._depth_events) - dropped_events)
        dropped_slices = int(len(items))
        self._dropped_slices += dropped_slices
        self._dropped_events += int(dropped_events)
        self._drop_counts[str(drop_reason)] += dropped_slices
        self._refresh_stats(str(drop_reason))
        return None

    def pop_ready(self, mask_filter, now_wall_us):
        if not self.enabled:
            return None
        if self._ready:
            item, reason, forced = self._ready.popleft()
            return self._release_item(item, reason, forced, mask_filter, now_wall_us)
        sync_i, bucket = self._oldest_bucket()
        if bucket is None:
            self._refresh_stats('empty')
            return None
        item = bucket['items'][0]
        age_us = max(0, int(now_wall_us or 0) - int(item.get('wall_time_us') or 0))

        try:
            if hasattr(mask_filter, 'poll_records_until_sync'):
                mask_filter.poll_records_until_sync(sync_i, force=False)
            else:
                mask_filter.poll_records(force=False)
            has_same_sync_mask = bool(mask_filter.has_record_for_sync(sync_i))
        except Exception:
            has_same_sync_mask = False
        due_budget = bool(age_us >= self.buffer_us)
        due_timeout = bool(age_us >= self.timeout_us)
        force_memory = bool(self._depth_events > self.max_events or len(self._buckets) > self.max_buckets)
        if self.strict_same_sync:
            if due_timeout:
                return self._drop_bucket_items(bucket, 'mask_timeout_drop')
            if has_same_sync_mask:
                return self._release_oldest_bucket_item(bucket, 'same_sync_ready', False,
                                                        mask_filter, now_wall_us)
            if force_memory:
                try:
                    if hasattr(mask_filter, 'poll_records_until_sync'):
                        mask_filter.poll_records_until_sync(sync_i, force=True)
                    else:
                        mask_filter.poll_records(force=True)
                    if mask_filter.has_record_for_sync(sync_i):
                        return self._release_oldest_bucket_item(bucket, 'same_sync_ready_after_force_poll', False,
                                                                mask_filter, now_wall_us)
                except Exception:
                    pass
                reason = 'max_events_drop' if self._depth_events > self.max_events else 'max_buckets_drop'
                return self._drop_bucket_items(bucket, reason)
            if due_budget:
                try:
                    if hasattr(mask_filter, 'poll_records_until_sync'):
                        mask_filter.poll_records_until_sync(sync_i, force=True)
                    else:
                        mask_filter.poll_records(force=True)
                    if mask_filter.has_record_for_sync(sync_i):
                        return self._release_oldest_bucket_item(bucket, 'same_sync_ready_after_budget_poll', False,
                                                                mask_filter, now_wall_us)
                except Exception:
                    pass
            self._refresh_stats('waiting_strict_same_sync')
            return None

        if force_memory:
            try:
                if hasattr(mask_filter, 'poll_records_until_sync'):
                    mask_filter.poll_records_until_sync(sync_i, force=True)
                else:
                    mask_filter.poll_records(force=True)
                if mask_filter.has_record_for_sync(sync_i):
                    return self._release_bucket_items(bucket, 'same_sync_ready_after_force_poll', False,
                                                      mask_filter, now_wall_us)
            except Exception:
                pass
            reason = 'max_events_force_raw' if self._depth_events > self.max_events else 'max_buckets_force_raw'
            return self._release_bucket_items(bucket, reason, True, mask_filter, now_wall_us)

        if due_budget and (self.release_policy == 'raw_at_budget' or due_timeout):
            try:
                if hasattr(mask_filter, 'poll_records_until_sync'):
                    mask_filter.poll_records_until_sync(sync_i, force=True)
                else:
                    mask_filter.poll_records(force=True)
                if mask_filter.has_record_for_sync(sync_i):
                    return self._release_bucket_items(bucket, 'same_sync_ready_after_budget_poll', False,
                                                      mask_filter, now_wall_us)
            except Exception:
                pass
            reason = 'timeout_raw' if due_timeout else 'budget_raw'
            return self._release_bucket_items(bucket, reason, False, mask_filter, now_wall_us)

        self._refresh_stats('waiting')
        return None

    def _refresh_stats(self, reason):
        oldest_age_ms = 0.0
        buffered_bucket_slices = 0
        if self._buckets:
            try:
                _, bucket = self._oldest_bucket()
                if bucket is not None and bucket.get('items'):
                    oldest_wall_us = int(bucket['items'][0].get('wall_time_us') or 0)
                    if oldest_wall_us > 0:
                        oldest_age_ms = max(0.0, (int(time.time() * 1_000_000) - oldest_wall_us) / 1000.0)
            except Exception:
                oldest_age_ms = 0.0
            try:
                buffered_bucket_slices = sum(len(bucket.get('items') or []) for bucket in self._buckets.values())
            except Exception:
                buffered_bucket_slices = 0
        queued_ready_slices = int(len(self._ready))
        legacy_held_slices = int(max(0, self._submitted_slices - self._released_slices - len(self._ready)))
        self._last_stats = {
            'enabled': bool(self.enabled),
            'reason': str(reason),
            'buffer_ms': float(self.buffer_ms),
            'timeout_ms': float(self.timeout_ms),
            'release_policy': str(self.release_policy),
            'mask_gate_mode': 'strict_same_sync' if self.strict_same_sync else 'fallback_raw',
            'raw_fallback_enabled': not bool(self.strict_same_sync),
            'depth_events': int(self._depth_events),
            'depth_buckets': int(len(self._buckets)),
            'submitted_slices': int(self._submitted_slices),
            'released_slices': int(self._released_slices),
            'dropped_slices': int(self._dropped_slices),
            'dropped_events': int(self._dropped_events),
            'held_slices': int(buffered_bucket_slices),
            'legacy_held_slices': int(legacy_held_slices),
            'buffered_bucket_slices': int(buffered_bucket_slices),
            'queued_ready_slices': int(queued_ready_slices),
            'logical_held_slices': int(buffered_bucket_slices + queued_ready_slices),
            'empty_release_fps': float(self.empty_release_fps),
            'empty_held_slices': int(self._empty_held_slices),
            'empty_skipped_slices': int(self._empty_skipped_slices),
            'forced_releases': int(self._forced_releases),
            'max_depth_events': int(self._max_depth_events),
            'max_depth_buckets': int(self._max_depth_buckets),
            'oldest_age_ms': round(float(oldest_age_ms), 3),
            'release_counts': dict(self._release_counts),
            'drop_counts': dict(self._drop_counts),
        }

    def stats(self):
        self._refresh_stats(getattr(self, '_last_stats', {}).get('reason', 'stats'))
        return dict(self._last_stats)


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


class MvsSoftTriggerTimelineMapper:
    """Map event-camera local trigger edges back to the MVS soft-trigger timeline."""

    def __init__(self, args, camera_alias: str, camera_serial: str):
        self.args = args
        self.camera_alias = str(camera_alias or '')
        self.camera_serial = str(camera_serial or '')
        self.source = str(getattr(args, 'event_sync_source', 'event_local') or 'event_local').strip().lower()
        self.enabled = self.source in ('mvs_soft_trigger', 'mvs_timeline', 'mvs_soft_trigger_timeline')
        self.wall_guard_us = int(round(max(0.0, float(getattr(args, 'event_sync_wall_guard_ms', 20.0) or 20.0)) * 1000.0))
        self.jump_warn_frames = max(1, int(getattr(args, 'event_sync_offset_jump_warn_frames', 2) or 2))
        self.local_fallback_enable = bool(getattr(args, 'event_sync_local_fallback_enable', True))
        self.timeline_path = self._format_path(
            str(getattr(args, 'event_sync_timeline_jsonl', './sync_ipc/global_trigger_timeline.jsonl') or '')
        )
        self.map_path = self._format_path(
            str(getattr(args, 'event_sync_map_jsonl_template', './sync_ipc/event_trigger_map_{camera}.jsonl') or '')
        )
        self.health_path = self._format_path(
            str(getattr(args, 'event_sync_health_json_template', './sync_ipc/event_sync_health_{camera}.json') or '')
        )
        self.ring_path = self._format_path(
            str(getattr(args, 'event_sync_map_ring_json_template', './sync_ipc/event_sync_map_ring_{camera}.json') or '')
        )
        self.tailer = _JsonlTailerLite(self.timeline_path) if self.enabled and self.timeline_path else None
        self.timeline = collections.deque(maxlen=max(64, int(getattr(args, 'event_sync_timeline_cache_records', 512) or 512)))
        self.ring_records = collections.deque(maxlen=max(32, int(getattr(args, 'event_sync_map_ring_records', 512) or 512)))
        self.by_sync = {}
        self.offset = None
        self.lock_state = 'DISABLED' if not self.enabled else 'UNLOCKED'
        self.lock_reason = 'disabled' if not self.enabled else 'init'
        self.latest_map = {}
        self.records_loaded = 0
        self.map_count = 0
        self.fallback_count = 0
        self.offset_jump_count = 0
        self.missing_timeline_count = 0
        self.out_of_guard_count = 0
        self.sync_start_tick_id = 0
        self.direct_index_map = False
        self.last_mvs_trigger_to_event_edge_us = None
        self._map_fp = None
        self._last_health_write_us = 0
        self._last_ring_write_us = 0
        self.ring_write_interval_us = int(round(max(
            0.0,
            float(getattr(args, 'event_sync_map_ring_write_interval_ms', 100.0) or 100.0),
        ) * 1000.0))
        self.ring_write_count = 0
        self.ring_write_error = ''
        if self.enabled:
            if self.map_path:
                try:
                    d = os.path.dirname(os.path.abspath(self.map_path))
                    if d:
                        os.makedirs(d, exist_ok=True)
                    self._map_fp = open(self.map_path, 'w', encoding='utf-8', buffering=1)
                except Exception as exc:
                    print(f"[EventSyncMapper][{self.camera_alias}][WARN] map jsonl open failed: {exc}", flush=True)
                    self._map_fp = None
            print(
                f"[EventSyncMapper][{self.camera_alias}] enabled source={self.source} "
                f"timeline={self.timeline_path} wall_guard_ms={self.wall_guard_us / 1000.0:.1f} "
                f"ring={self.ring_path} ring_records={self.ring_records.maxlen}",
                flush=True,
            )

    def _format_path(self, raw_path: str) -> str:
        raw = str(raw_path or '').strip()
        if not raw:
            return ''
        try:
            raw = raw.format(
                camera=self.camera_alias,
                cam=self.camera_alias,
                alias=self.camera_alias,
                id=self.camera_alias,
                serial=self.camera_serial,
                camera_sn=self.camera_serial,
            )
        except Exception:
            pass
        return os.path.abspath(os.path.expanduser(raw))

    @staticmethod
    def _rec_sync(rec):
        for key in ('program_sync_index', 'trigger_seq', 'tick_id', 'sync_index'):
            try:
                if key in rec and rec.get(key) is not None:
                    return int(rec.get(key))
            except Exception:
                continue
        return None

    @staticmethod
    def _rec_wall_us(rec):
        for key in ('mvs_soft_trigger_est_wall_us', 'soft_trigger_fire_wall_us', 'coord_send_wall_us', 'wall_time_us'):
            try:
                v = rec.get(key)
                if v is not None:
                    return int(float(v))
            except Exception:
                continue
        return 0

    def poll_timeline(self, force: bool = False) -> int:
        if not self.enabled or self.tailer is None:
            return 0
        loaded = 0
        for rec in self.tailer.poll():
            sync_i = self._rec_sync(rec)
            if sync_i is None:
                continue
            item = dict(rec)
            item['program_sync_index'] = int(sync_i)
            item['_mvs_wall_us'] = int(self._rec_wall_us(item))
            self.by_sync[int(sync_i)] = item
            self.timeline.append(item)
            loaded += 1
        if len(self.by_sync) > self.timeline.maxlen:
            keep = {int(r.get('program_sync_index')) for r in self.timeline if r.get('program_sync_index') is not None}
            for key in list(self.by_sync.keys()):
                if key not in keep:
                    self.by_sync.pop(key, None)
        self.records_loaded += int(loaded)
        return loaded

    def reset_sync_start(self, tick_id_start=None, direct_index_map=False):
        try:
            self.sync_start_tick_id = int(tick_id_start if tick_id_start is not None else 0)
        except Exception:
            self.sync_start_tick_id = 0
        self.direct_index_map = bool(direct_index_map)
        self.offset = None
        self.lock_state = 'UNLOCKED' if self.enabled else 'DISABLED'
        if self.direct_index_map and self.enabled:
            self.lock_reason = f'file_replay_direct_index_map_start={self.sync_start_tick_id}'
        else:
            self.lock_reason = f'sync_start_tick_id={self.sync_start_tick_id}'

    def _nearest_by_wall(self, edge_wall_us):
        if edge_wall_us is None:
            return None, None
        best = None
        best_delta = None
        for rec in list(self.timeline):
            wall = int(rec.get('_mvs_wall_us') or 0)
            if wall <= 0:
                continue
            delta = int(edge_wall_us) - int(wall)
            if best is None or abs(delta) < abs(best_delta):
                best = rec
                best_delta = delta
        return best, best_delta

    @staticmethod
    def _edge_wall_us(event_ts_us, slice_ts_us, wall_time_us):
        try:
            return int(wall_time_us) + (int(event_ts_us) - int(slice_ts_us))
        except Exception:
            try:
                return int(wall_time_us)
            except Exception:
                return None

    def map_selected_trigger(self, *, local_sync_index, event_ts_us, slice_ts_us, wall_time_us, polarity, trigger_id):
        if not self.enabled:
            return {
                'sync_master': 'event_local',
                'mapped_mvs_sync_index': int(local_sync_index),
                'event_local_sync_index': int(local_sync_index),
                'mvs_soft_trigger_event_ts_us': None,
                'mvs_trigger_to_event_edge_us_used': None,
                'mvs_soft_trigger_event_ts_source': 'event_local',
                'lock_state': 'DISABLED',
                'reason': 'event_sync_source_event_local',
            }
        self.poll_timeline(force=False)
        local_i = int(local_sync_index)
        edge_wall = self._edge_wall_us(event_ts_us, slice_ts_us, wall_time_us)
        mapped = None
        reason = ''
        match_mode = ''
        nearest = None
        delta_us = None
        nearest_sync = None
        if bool(getattr(self, 'direct_index_map', False)):
            mapped = int(local_i) + int(self.sync_start_tick_id)
            self.offset = int(self.sync_start_tick_id)
            self.lock_state = 'FILE_REPLAY_DIRECT_INDEX'
            reason = 'file_replay_sync_start_direct_index'
            match_mode = 'sync_start_direct_index'
        else:
            nearest, delta_us = self._nearest_by_wall(edge_wall)
            nearest_sync = None if nearest is None else int(nearest.get('program_sync_index'))

        if (
            mapped is None
            and nearest_sync is not None
            and delta_us is not None
            and abs(int(delta_us)) <= self.wall_guard_us
        ):
            mapped = int(nearest_sync)
            new_offset = int(mapped) - int(local_i)
            if self.offset is None:
                self.offset = int(new_offset)
                self.lock_state = 'LOCKED'
                reason = 'timeline_wall_lock'
            elif abs(int(new_offset) - int(self.offset)) <= self.jump_warn_frames:
                self.lock_state = 'LOCKED'
                reason = 'timeline_wall_confirm'
            else:
                self.offset_jump_count += 1
                self.offset = int(new_offset)
                self.lock_state = 'DEGRADED'
                reason = 'timeline_offset_jump_relock'
            match_mode = 'timeline_nearest_wall'
        elif mapped is None:
            if nearest_sync is None:
                self.missing_timeline_count += 1
                reason = 'timeline_missing'
            else:
                self.out_of_guard_count += 1
                reason = f'timeline_wall_out_of_guard:dt_ms={float(delta_us or 0) / 1000.0:.3f}'
            if self.offset is not None:
                mapped = int(local_i) + int(self.offset)
                self.lock_state = 'LOCKED_OFFSET'
                match_mode = 'offset_hold'
            elif self.local_fallback_enable:
                mapped = int(local_i) + int(self.sync_start_tick_id)
                self.lock_state = 'LOCAL_FALLBACK'
                self.fallback_count += 1
                match_mode = 'sync_start_local_fallback'
            else:
                mapped = None
                self.lock_state = 'UNLOCKED'

        delay_used = None
        anchor_ts_us = None
        anchor_source = ''
        if match_mode == 'sync_start_direct_index':
            delay_used = 0
            anchor_source = 'file_replay_direct_index'
        elif match_mode == 'timeline_nearest_wall' and delta_us is not None:
            delay_used = int(delta_us)
            self.last_mvs_trigger_to_event_edge_us = int(delay_used)
            anchor_source = 'timeline_wall_delta'
        elif mapped is not None and self.last_mvs_trigger_to_event_edge_us is not None:
            delay_used = int(self.last_mvs_trigger_to_event_edge_us)
            anchor_source = 'held_wall_delta'
        elif mapped is not None:
            delay_used = 0
            anchor_source = 'event_edge_fallback'
        if delay_used is not None:
            anchor_ts_us = int(event_ts_us) - int(delay_used)

        self.map_count += 1
        self.lock_reason = reason
        rec = {
            'msg_type': 'event_trigger_map',
            'camera': self.camera_alias,
            'sync_master': 'mvs_soft_trigger',
            'event_local_sync_index': int(local_i),
            'mapped_mvs_sync_index': None if mapped is None else int(mapped),
            'nearest_mvs_sync_index': None if nearest_sync is None else int(nearest_sync),
            'event_edge_ts_us': int(event_ts_us),
            'event_edge_wall_us_est': None if edge_wall is None else int(edge_wall),
            'slice_ts_us': int(slice_ts_us or 0),
            'slice_wall_us': int(wall_time_us or 0),
            'mvs_trigger_to_event_edge_us': None if delta_us is None else int(delta_us),
            'mvs_trigger_to_event_edge_us_used': None if delay_used is None else int(delay_used),
            'mvs_soft_trigger_event_ts_us': None if anchor_ts_us is None else int(anchor_ts_us),
            'mvs_soft_trigger_event_ts_source': str(anchor_source),
            'offset': None if self.offset is None else int(self.offset),
            'lock_state': str(self.lock_state),
            'match_mode': str(match_mode),
            'reason': str(reason),
            'polarity': int(polarity),
            'trigger_id': int(trigger_id),
            'timeline_records_loaded': int(self.records_loaded),
            'direct_index_map': bool(getattr(self, 'direct_index_map', False)),
            'rgb_exposure_as_sync_anchor': False,
        }
        self.latest_map = dict(rec)
        if self._map_fp is not None:
            try:
                self._map_fp.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
            except Exception:
                pass
        self._append_ring_record(rec)
        self._maybe_write_health()
        return rec

    def _append_ring_record(self, rec):
        if not isinstance(rec, dict):
            return
        item = dict(rec)
        item['ring_record_wall_us'] = int(_now_wall_time_us())
        self.ring_records.append(item)
        self._maybe_write_ring(force=False)

    def _maybe_write_ring(self, force: bool = False):
        if not self.ring_path:
            return
        now_us_i = int(_now_wall_time_us())
        if not force and self.ring_write_interval_us > 0:
            if now_us_i - int(self._last_ring_write_us or 0) < int(self.ring_write_interval_us):
                return
        self._last_ring_write_us = now_us_i
        payload = {
            'schema_version': 1,
            'source': 'event_sync_map_ring',
            'camera': self.camera_alias,
            'sync_master': 'mvs_soft_trigger' if self.enabled else 'event_local',
            'published_wall_us': int(now_us_i),
            'enabled': bool(self.enabled),
            'lock_state': str(self.lock_state),
            'lock_reason': str(self.lock_reason),
            'offset': None if self.offset is None else int(self.offset),
            'records_written': int(self.map_count),
            'records_kept': int(len(self.ring_records)),
            'records_max': int(self.ring_records.maxlen or 0),
            'timeline_records_loaded': int(self.records_loaded),
            'direct_index_map': bool(getattr(self, 'direct_index_map', False)),
            'fallback_count': int(self.fallback_count),
            'offset_jump_count': int(self.offset_jump_count),
            'missing_timeline_count': int(self.missing_timeline_count),
            'out_of_guard_count': int(self.out_of_guard_count),
            'latest_map': dict(self.latest_map or {}),
            'records': list(self.ring_records),
            'write_count': int(self.ring_write_count),
            'last_error': str(self.ring_write_error),
        }
        try:
            d = os.path.dirname(os.path.abspath(self.ring_path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.ring_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
            os.replace(tmp, self.ring_path)
            self.ring_write_count += 1
            self.ring_write_error = ''
        except Exception as exc:
            self.ring_write_error = f'{type(exc).__name__}: {exc}'

    def _maybe_write_health(self):
        now_us_i = int(_now_wall_time_us())
        if now_us_i - int(self._last_health_write_us or 0) < 200_000:
            return
        self._last_health_write_us = now_us_i
        if not self.health_path:
            return
        payload = self.health()
        try:
            d = os.path.dirname(os.path.abspath(self.health_path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.health_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
            os.replace(tmp, self.health_path)
        except Exception:
            pass

    def health(self):
        out = {
            'enabled': bool(self.enabled),
            'sync_master': 'mvs_soft_trigger' if self.enabled else 'event_local',
            'source': str(self.source),
            'camera': self.camera_alias,
            'lock_state': str(self.lock_state),
            'lock_reason': str(self.lock_reason),
            'offset': None if self.offset is None else int(self.offset),
            'sync_start_tick_id': int(self.sync_start_tick_id),
            'map_count': int(self.map_count),
            'fallback_count': int(self.fallback_count),
            'offset_jump_count': int(self.offset_jump_count),
            'missing_timeline_count': int(self.missing_timeline_count),
            'out_of_guard_count': int(self.out_of_guard_count),
            'timeline_records_loaded': int(self.records_loaded),
            'direct_index_map': bool(getattr(self, 'direct_index_map', False)),
            'timeline_path': str(self.timeline_path),
            'ring_path': str(self.ring_path),
            'ring_records_kept': int(len(self.ring_records)),
            'ring_records_max': int(self.ring_records.maxlen or 0),
            'ring_write_count': int(self.ring_write_count),
            'ring_write_error': str(self.ring_write_error),
            'wall_guard_ms': float(self.wall_guard_us) / 1000.0,
            'latest_map': dict(self.latest_map or {}),
            'rgb_exposure_as_sync_anchor': False,
            'wall_time_us': int(_now_wall_time_us()),
        }
        return out

    def close(self):
        try:
            self._maybe_write_ring(force=True)
            if self._map_fp is not None:
                self._map_fp.flush()
                self._map_fp.close()
        except Exception:
            pass
        self._map_fp = None


class EventReplayMvsPacer:
    """Backpressure file-replayed Event slices against the MVS replay tick.

    The native Event file broker can play a RAW at real-time speed while the MVS
    file broker is slower because it decodes and republishes eight videos.  In
    direct-index replay that makes the Event producer head drift ahead of human
    masks, hit evidence, and BEV target sync.  A small wait here naturally
    backpressures the FIFO without affecting live HAL mode.
    """

    def __init__(self, args):
        self.enabled = bool(getattr(args, 'event_replay_mvs_pacing_enable', True))
        self.lookahead_ticks = max(0, int(getattr(args, 'event_replay_mvs_pacing_lookahead_ticks', 2) or 0))
        self.max_wait_ms = max(0.0, float(getattr(args, 'event_replay_mvs_pacing_max_wait_ms', 250.0) or 0.0))
        self.poll_ms = max(1.0, float(getattr(args, 'event_replay_mvs_pacing_poll_ms', 5.0) or 5.0))
        self.status_path = os.path.abspath(os.path.expanduser(
            str(getattr(args, 'event_replay_mvs_pacing_trigger_status_json', './sync_ipc/trigger_status.json') or '')
        ))
        self._last_read_monotonic = 0.0
        self._cached_target = None
        self._cached_payload = {}
        self.wait_count = 0
        self.timeout_count = 0
        self.total_wait_ms = 0.0
        self.max_observed_ahead_ticks = 0
        self.last_diag = {
            'enabled': bool(self.enabled),
            'state': 'init' if self.enabled else 'disabled',
            'status_path': self.status_path,
            'lookahead_ticks': int(self.lookahead_ticks),
            'max_wait_ms': float(self.max_wait_ms),
        }

    @staticmethod
    def _payload_tick(payload):
        if not isinstance(payload, dict):
            return None
        for key in ('tick_id', 'program_sync_index', 'sync_index', 'trigger_seq'):
            try:
                value = payload.get(key)
                if value is not None and value != '':
                    return int(float(value))
            except Exception:
                continue
        return None

    def _read_target(self, force=False):
        if not self.enabled or not self.status_path:
            return None, {}
        now_m = time.monotonic()
        if not force and self._cached_target is not None and (now_m - self._last_read_monotonic) < 0.02:
            return self._cached_target, self._cached_payload
        self._last_read_monotonic = now_m
        try:
            with open(self.status_path, 'r', encoding='utf-8', errors='replace') as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                payload = {}
        except Exception as exc:
            self._cached_payload = {'read_error': f'{type(exc).__name__}: {exc}'}
            self._cached_target = None
            return None, self._cached_payload
        target = self._payload_tick(payload)
        self._cached_payload = payload
        self._cached_target = target
        return target, payload

    def maybe_wait(self, mapped_sync_index, *, direct_index_map=False):
        if not self.enabled:
            self.last_diag = {'enabled': False, 'state': 'disabled'}
            return self.last_diag
        if not bool(direct_index_map):
            self.last_diag = {
                'enabled': True,
                'state': 'bypass_not_direct_index',
                'status_path': self.status_path,
            }
            return self.last_diag
        try:
            mapped_i = int(mapped_sync_index)
        except Exception:
            self.last_diag = {
                'enabled': True,
                'state': 'bypass_no_mapped_sync',
                'status_path': self.status_path,
            }
            return self.last_diag

        target, payload = self._read_target(force=False)
        if target is None:
            self.last_diag = {
                'enabled': True,
                'state': 'no_mvs_target',
                'mapped_sync_index': int(mapped_i),
                'status_path': self.status_path,
                'read_error': self._cached_payload.get('read_error') if isinstance(self._cached_payload, dict) else '',
            }
            return self.last_diag

        ahead_before = int(mapped_i) - int(target)
        self.max_observed_ahead_ticks = max(int(self.max_observed_ahead_ticks), int(ahead_before))
        if ahead_before <= int(self.lookahead_ticks):
            self.last_diag = {
                'enabled': True,
                'state': 'in_window',
                'mapped_sync_index': int(mapped_i),
                'target_sync_index': int(target),
                'ahead_ticks': int(ahead_before),
                'lookahead_ticks': int(self.lookahead_ticks),
                'source_mode': str(payload.get('source_mode', '')),
            }
            return self.last_diag

        start = time.monotonic()
        deadline = start + (self.max_wait_ms / 1000.0)
        waited_ms = 0.0
        timed_out = False
        while True:
            now_m = time.monotonic()
            if now_m >= deadline:
                timed_out = True
                break
            sleep_s = min(max(0.001, self.poll_ms / 1000.0), max(0.0, deadline - now_m))
            time.sleep(sleep_s)
            target, payload = self._read_target(force=True)
            if target is None:
                break
            if int(mapped_i) - int(target) <= int(self.lookahead_ticks):
                break
        waited_ms = (time.monotonic() - start) * 1000.0
        target_after = target
        ahead_after = None if target_after is None else int(mapped_i) - int(target_after)
        self.wait_count += 1
        self.total_wait_ms += float(waited_ms)
        if timed_out:
            self.timeout_count += 1
        self.last_diag = {
            'enabled': True,
            'state': 'timeout_release' if timed_out else 'waited',
            'mapped_sync_index': int(mapped_i),
            'target_sync_index': None if target_after is None else int(target_after),
            'ahead_ticks_before': int(ahead_before),
            'ahead_ticks_after': ahead_after,
            'lookahead_ticks': int(self.lookahead_ticks),
            'waited_ms': round(float(waited_ms), 3),
            'wait_count': int(self.wait_count),
            'timeout_count': int(self.timeout_count),
            'total_wait_ms': round(float(self.total_wait_ms), 3),
            'max_observed_ahead_ticks': int(self.max_observed_ahead_ticks),
            'status_path': self.status_path,
            'source_mode': str(payload.get('source_mode', '')) if isinstance(payload, dict) else '',
        }
        return self.last_diag

    def stats(self):
        out = dict(self.last_diag or {})
        out.update({
            'enabled': bool(self.enabled),
            'status_path': self.status_path,
            'lookahead_ticks': int(self.lookahead_ticks),
            'max_wait_ms': float(self.max_wait_ms),
            'poll_ms': float(self.poll_ms),
            'wait_count': int(self.wait_count),
            'timeout_count': int(self.timeout_count),
            'total_wait_ms': round(float(self.total_wait_ms), 3),
            'max_observed_ahead_ticks': int(self.max_observed_ahead_ticks),
        })
        return out


def _drain_external_triggers(mv_iterator, args, writer, state, slice_ts_us, wall_time_us, sync_mapper=None):
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

        # PATCH ext-trigger selected-edge refractory begin
        selected = (int(p) == selected_polarity)
        dedupe_us = int(round(max(0.0, float(getattr(args, 'ext_trigger_dedupe_ms', 18.0))) * 1000.0))
        if selected and dedupe_us > 0:
            last_accept_t = state.get('last_accepted_selected_event_ts_us')
            if last_accept_t is not None:
                dt_us = int(t) - int(last_accept_t)
                if 0 <= dt_us < dedupe_us:
                    state['dedupe_skipped_edges'] = int(state.get('dedupe_skipped_edges', 0) or 0) + 1
                    continue
            state['last_accepted_selected_event_ts_us'] = int(t)
        # PATCH ext-trigger selected-edge refractory end

        all_index = state['all_index']
        state['all_index'] += 1

        sync_index = -1
        sync_map = None
        if selected:
            sync_index = state['sync_index']
            state['sync_index'] += 1
            state['last_selected_sync_index'] = int(sync_index)
            state['last_selected_event_ts_us'] = int(t)
            if sync_mapper is not None:
                try:
                    sync_map = sync_mapper.map_selected_trigger(
                        local_sync_index=int(sync_index),
                        event_ts_us=int(t),
                        slice_ts_us=int(slice_ts_us),
                        wall_time_us=int(wall_time_us),
                        polarity=int(p),
                        trigger_id=int(tid),
                    )
                except Exception as exc:
                    sync_map = {
                        'event_local_sync_index': int(sync_index),
                        'mapped_mvs_sync_index': None,
                        'mvs_soft_trigger_event_ts_us': None,
                        'mvs_trigger_to_event_edge_us_used': None,
                        'mvs_soft_trigger_event_ts_source': 'mapper_error',
                        'lock_state': 'MAPPER_ERROR',
                        'reason': f'{type(exc).__name__}: {exc}',
                    }
            if isinstance(sync_map, dict):
                state['last_event_sync_map'] = dict(sync_map)
                mapped_sync = sync_map.get('mapped_mvs_sync_index')
                if mapped_sync is not None:
                    try:
                        state['last_mapped_mvs_sync_index'] = int(mapped_sync)
                        state['last_mapped_mvs_sync_event_ts_us'] = int(t)
                    except Exception:
                        pass
            if print_every > 0 and state['sync_index'] % print_every == 0:
                print(
                    f"[SYNC] selected trigger sync_index={sync_index} "
                    f"mapped_mvs={None if not isinstance(sync_map, dict) else sync_map.get('mapped_mvs_sync_index')} "
                    f"lock={None if not isinstance(sync_map, dict) else sync_map.get('lock_state')} "
                    f"event_ts_us={int(t)} polarity={int(p)} id={int(tid)}",
                    flush=True,
                )

        if writer is not None and (selected or bool(getattr(args, 'ext_trigger_log_all_edges', False))):
            writer.writerow({
                'all_index': int(all_index),
                'sync_index': int(sync_index),
                'event_ts_us': int(t),
                'polarity': int(p),
                'trigger_id': int(tid),
                'slice_ts_us': int(slice_ts_us),
                'wall_time_us': int(wall_time_us),
                'mapped_mvs_sync_index': '' if not isinstance(sync_map, dict) or sync_map.get('mapped_mvs_sync_index') is None else int(sync_map.get('mapped_mvs_sync_index')),
                'sync_master': '' if not isinstance(sync_map, dict) else str(sync_map.get('sync_master', '')),
                'event_local_sync_index': '' if not selected else int(sync_index),
                'mvs_soft_trigger_event_ts_us': '' if not isinstance(sync_map, dict) or sync_map.get('mvs_soft_trigger_event_ts_us') is None else int(sync_map.get('mvs_soft_trigger_event_ts_us')),
                'mvs_trigger_to_event_edge_us': '' if not isinstance(sync_map, dict) or sync_map.get('mvs_trigger_to_event_edge_us') is None else int(sync_map.get('mvs_trigger_to_event_edge_us')),
                'mvs_trigger_to_event_edge_us_used': '' if not isinstance(sync_map, dict) or sync_map.get('mvs_trigger_to_event_edge_us_used') is None else int(sync_map.get('mvs_trigger_to_event_edge_us_used')),
                'mvs_soft_trigger_event_ts_source': '' if not isinstance(sync_map, dict) else str(sync_map.get('mvs_soft_trigger_event_ts_source', '')),
                'sync_lock_state': '' if not isinstance(sync_map, dict) else str(sync_map.get('lock_state', '')),
                'sync_match_mode': '' if not isinstance(sync_map, dict) else str(sync_map.get('match_mode', '')),
            })
            written_count += 1
        elif writer is not None and not selected:
            state['nonselected_log_skipped_edges'] = int(state.get('nonselected_log_skipped_edges', 0) or 0) + 1

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


def _activate_ext_trigger_sync_start_if_ready(mv_iterator, args, state, ext_trigger_file, ext_trigger_writer,
                                              camera_alias: str, sync_mapper=None) -> bool:
    """Activate formal sync origin when sync_start.flag appears."""
    if not bool(state.get('sync_gate_required', False)):
        return False
    if bool(state.get('sync_started', False)):
        return False
    path = str(state.get('sync_start_path', '') or '')
    if not path or not os.path.exists(path):
        return False

    sync_start_payload = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            sync_start_payload = json.loads(f.readline().strip() or '{}')
            if not isinstance(sync_start_payload, dict):
                sync_start_payload = {}
    except Exception:
        sync_start_payload = {}

    _clear_ext_trigger_events_from_iterator(mv_iterator)

    state['all_index'] = 0
    state['sync_index'] = 0
    state['last_selected_sync_index'] = None
    state['last_selected_event_ts_us'] = None
    try:
        state['sync_start_tick_id'] = int(sync_start_payload.get('tick_id_start', 0) or 0)
    except Exception:
        state['sync_start_tick_id'] = 0
    direct_index_map = False
    try:
        direct_index_map = bool(sync_start_payload.get('event_sync_direct_index_map', False))
        source_mode = str(sync_start_payload.get('source_mode', '') or '').strip().lower()
        if source_mode in ('mvs_file_replay', 'file_replay'):
            direct_index_map = True
    except Exception:
        direct_index_map = False
    state['event_sync_direct_index_map'] = bool(direct_index_map)
    # PATCH ext-trigger sync reset begin
    state['last_accepted_selected_event_ts_us'] = None
    state['dedupe_skipped_edges'] = 0
    # PATCH ext-trigger sync reset end
    try:
        state['seen'].clear()
    except Exception:
        state['seen'] = set()
    state['sync_started'] = True
    state['sync_start_wall_time_us'] = int(_now_wall_time_us())
    if sync_mapper is not None:
        try:
            sync_mapper.reset_sync_start(
                state.get('sync_start_tick_id', 0),
                direct_index_map=bool(direct_index_map),
            )
        except Exception:
            pass

    if bool(getattr(args, 'ext_trigger_sync_start_reset_log',
                    False)) and ext_trigger_file is not None and ext_trigger_writer is not None:
        try:
            ext_trigger_file.seek(0)
            ext_trigger_file.truncate(0)
            ext_trigger_writer.writeheader()
            ext_trigger_file.flush()
        except Exception as exc:
            print(f"[SYNC][{camera_alias}][WARN] reset event_trigger CSV on sync_start failed: {exc}", flush=True)

    print(
        f"[SYNC][{camera_alias}] formal sync_start detected -> reset EVENT sync_index=0, "
        f"mvs_tick_start={state.get('sync_start_tick_id', 0)}, "
        f"direct_index_map={bool(direct_index_map)}, path={path}",
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
        '--server-ip', args.tsf1_server_ip,
        '--server-port', str(args.tsf1_server_port),
        '--camera-id', args.tsf1_camera_id,
        '--send-fps', str(args.tsf1_pub_fps),
        '--codec', args.tsf1_codec,
        '--zlib-level', str(args.tsf1_zlib_level),
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


def _publish_shared_bgr_frame(writer, frame_bgr, timestamp_us: int) -> dict:
    if writer is None or frame_bgr is None:
        return {}
    try:
        frame = np.asarray(frame_bgr)
        writer_channels = int(getattr(writer, 'channels', 3) or 3)
        if writer_channels == 1:
            if frame.ndim == 3:
                if frame.shape[2] == 1:
                    frame = frame[:, :, 0]
                else:
                    frame = np.max(frame[:, :, :3], axis=2)
            if frame.shape[:2] != (writer.height, writer.width):
                frame = cv2.resize(frame, (writer.width, writer.height), interpolation=cv2.INTER_AREA)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            lum = frame
            writer.write(frame, timestamp_us=timestamp_us)
            out_h, out_w = frame.shape[:2]
        else:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            if frame.shape[:2] != (writer.height, writer.width):
                frame = cv2.resize(frame, (writer.width, writer.height), interpolation=cv2.INTER_AREA)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            lum = np.max(frame[:, :, :3], axis=2) if frame.ndim == 3 else frame
            writer.write(frame, timestamp_us=timestamp_us)
            out_h, out_w = frame.shape[:2]
        nonzero_pixels = int(np.count_nonzero(lum))
        max_luma = int(np.max(lum)) if lum.size else 0
        mean_luma = float(np.mean(lum)) if lum.size else 0.0
        return {
            'shm_frame_id': int(getattr(writer, '_frame_id', -1)),
            'shm_timestamp_us': int(timestamp_us or 0),
            'overlay_nonzero_pixels': int(nonzero_pixels),
            'overlay_max_luma': int(max_luma),
            'overlay_mean_luma': float(mean_luma),
            'overlay_width': int(getattr(writer, 'width', out_w)),
            'overlay_height': int(getattr(writer, 'height', out_h)),
            'overlay_channels': int(writer_channels),
        }
    except Exception as exc:
        now = time.time()
        last = float(getattr(writer, '_last_overlay_pub_warn_wall', 0.0))
        if now - last > 5.0:
            setattr(writer, '_last_overlay_pub_warn_wall', now)
            print(f'[EventOverlayPub][WARN] publish failed: {type(exc).__name__}: {exc}', flush=True)
        return {'overlay_publish_error': f'{type(exc).__name__}: {exc}'}


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
        return datetime.datetime.fromtimestamp(int(wall_time_us) / 1_000_000.0).astimezone().isoformat(
            timespec='microseconds')
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


def _update_spatter_summary(stats: dict, ts, raw_n, cluster_count, accepted_count, interval_ms: int,
                            camera_log_tag: str, camera_sn_for_log: str):
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
            f"dt_ms={(int(ts) - int(stats['start_ts'])) / 1000.0:.1f} "
            f"slices={slices} raw_avg={stats['raw_events'] / slices:.1f} "
            f"cluster_avg={stats['clusters'] / slices:.2f} cluster_max={stats['max_clusters']} "
            f"zero_slices={stats['zero_cluster_slices']}/{slices} "
            f"accepted_avg={stats['accepted'] / slices:.2f} accepted_max={stats['max_accepted']}",
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
    ts_i = int(ts)
    fg_state_all = getattr(_maybe_render, '_frame_gen_state', None)
    if not isinstance(fg_state_all, dict):
        fg_state_all = {}
        setattr(_maybe_render, '_frame_gen_state', fg_state_all)
    fg_state = fg_state_all.setdefault(int(id(events_frame_gen_algo)), {})
    last_ts = fg_state.get('last_generate_ts_us')
    generate_ts_i = ts_i
    if last_ts is not None and ts_i < int(last_ts):
        generate_ts_i = int(last_ts)
        clamp_cnt = int(fg_state.get('backward_clamp_count', 0) or 0) + 1
        fg_state['backward_clamp_count'] = clamp_cnt
        fg_state['last_generate_requested_ts_us'] = ts_i
        if hasattr(events_frame_gen_algo, 'reset'):
            try:
                events_frame_gen_algo.reset()
                reset_cnt = int(fg_state.get('backward_reset_count', 0) or 0) + 1
                fg_state['backward_reset_count'] = reset_cnt
                if reset_cnt <= 5 or reset_cnt % 100 == 0:
                    print(
                        f"[EventViz][WARN] frame timestamp moved backward; reset renderer "
                        f"last_ts={int(last_ts)} current_ts={ts_i} count={reset_cnt}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[EventViz][WARN] reset before backward frame failed: {type(exc).__name__}: {exc}", flush=True)
        if clamp_cnt <= 5 or clamp_cnt % 100 == 0:
            print(
                f"[EventViz][WARN] clamp frame timestamp rollback "
                f"requested_ts={ts_i} generate_ts={generate_ts_i} last_ts={int(last_ts)} count={clamp_cnt}",
                flush=True,
            )
    generated = False
    try:
        events_frame_gen_algo.generate(generate_ts_i, output_img)
        generated = True
    except ValueError as exc:
        msg = str(exc)
        if 'past' in msg.lower() and hasattr(events_frame_gen_algo, 'reset'):
            try:
                events_frame_gen_algo.reset()
                cnt = int(fg_state.get('backward_retry_count', 0) or 0) + 1
                fg_state['backward_retry_count'] = cnt
                m = re.search(r"Last frame was generated at\s+([0-9]+)", msg)
                internal_last = int(m.group(1)) if m else generate_ts_i
                retry_ts_i = max(generate_ts_i, internal_last)
                print(
                    f"[EventViz][WARN] generate timestamp rollback; reset and retry once "
                    f"requested_ts={ts_i} generate_ts={generate_ts_i} retry_ts={retry_ts_i}: {msg}",
                    flush=True,
                )
                events_frame_gen_algo.generate(retry_ts_i, output_img)
                generate_ts_i = retry_ts_i
                generated = True
            except Exception as retry_exc:
                cnt = int(fg_state.get('backward_skip_count', 0) or 0) + 1
                fg_state['backward_skip_count'] = cnt
                fg_state['last_generate_error'] = f"{type(retry_exc).__name__}: {retry_exc}"
                if hasattr(output_img, 'fill'):
                    try:
                        output_img.fill(0)
                    except Exception:
                        pass
                if cnt <= 5 or cnt % 100 == 0:
                    print(
                        f"[EventViz][WARN] skip event frame after timestamp rollback retry failed "
                        f"requested_ts={ts_i} generate_ts={generate_ts_i} count={cnt}: {type(retry_exc).__name__}: {retry_exc}",
                        flush=True,
                    )
        else:
            raise
    if generated:
        fg_state['last_generate_ts_us'] = generate_ts_i

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
    args = parse_args()
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
    _sender_camera_alias = str(getattr(args, 'hit_camera_alias', '') or '').strip()
    _sender_camera_name = _sender_camera_alias or str(getattr(args, 'tsf1_camera_id', '') or '').strip() or 'cam'
    bullet_sender = BulletEventSender(
        server_ip=args.hit_server_ip or args.tsf1_server_ip,
        server_port=args.hit_server_port,
        enabled=bool(args.enable_hit_events and (args.hit_server_ip or args.tsf1_server_ip)),
        camera_alias=_sender_camera_alias,
        name=_sender_camera_name,
    )
    bullet_extractor = BulletEventExtractor(
        sender=bullet_sender,
        camera_sn=args.tsf1_camera_id,  # --tsf1-camera-id 填序列号即作为全链路唯一主键
        camera_alias=getattr(args, 'hit_camera_alias', ''),
        orig_w=width,
        orig_h=height,
        min_step_px=args.line_min_step_px,
    )
    bullet_sender.start()
    bullet_display_diag = {
        'draw_paths_count': 0,
        'draw_path_points_count': 0,
        'display_eligible_count': 0,
        'active_track_count': 0,
        'active_but_not_display_count': 0,
        'last_overlay_bullet_point_age_ms': -1.0,
        'bullet_sender_queue_size': 0,
        'bullet_sender_dropped': 0,
        'bullet_sender_camera_alias': _sender_camera_alias,
        'bullet_sender_name': _sender_camera_name,
        'visual_point_submit_count': 0,
        'visual_point_submit_failed_count': 0,
        'visual_point_skip_count': 0,
        'bullet_source_state': 'init',
    }

    runtime_switches = {
        'live_mode': live,
        'stc_enabled': args.stc_threshold > 0,
        'tsf1_pub_enabled': args.enable_tsf1_pub,
        'event_overlay_pub_enabled': bool(getattr(args, 'enable_event_overlay_pub', False)),
        'replay_wrapped': args.replay_factor > 0 and not live,
        'headless': args.headless,
        'render_every_n': max(1, args.render_every_n),
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
    clusters = EventSpatterClusterBuffer()

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
    _apply_event_worker_affinity(args, camera_short_alias)
    camera_sn_for_log = str(getattr(args, 'tsf1_camera_id', '') or '').strip()
    camera_log_tag = camera_short_alias if camera_short_alias else camera_sn_for_log if camera_sn_for_log else 'cam'
    window_title = _get_window_title(args)
    hot_pixel_filter = EventHotPixelFilter(args, camera_short_alias)

    def _build_shadow_line_filter():
        cams_spec = str(getattr(args, 'event_line_shadow_cameras', '') or '').strip()
        if not bool(getattr(args, 'event_line_shadow_enable', False)):
            return None, {'enabled': False, 'reason': 'disabled'}
        enabled_cams = {x.strip().upper() for x in re.split(r'[,;\s]+', cams_spec) if x.strip()}
        cam_alias = str(camera_short_alias or '').strip().upper()
        if enabled_cams and cam_alias not in enabled_cams and 'ALL' not in enabled_cams and '*' not in enabled_cams:
            return None, {'enabled': False, 'reason': 'camera_not_selected', 'cameras': sorted(enabled_cams)}
        shadow_args = copy.copy(args)
        overrides = {}
        v = float(getattr(args, 'event_line_shadow_min_step_px', -1.0) or -1.0)
        if v > 0:
            shadow_args.line_min_step_px = v
            overrides['line_min_step_px'] = v
        v = float(getattr(args, 'event_line_shadow_min_total_disp_px', -1.0) or -1.0)
        if v > 0:
            shadow_args.line_min_total_disp_px = v
            overrides['line_min_total_disp_px'] = v
        iv = int(getattr(args, 'event_line_shadow_bullet_min_output_streak', -1) or -1)
        if iv > 0:
            shadow_args.bullet_min_output_streak = iv
            overrides['bullet_min_output_streak'] = iv
        iv = int(getattr(args, 'event_line_shadow_untracked_threshold', -1) or -1)
        if iv > 0:
            shadow_args.untracked_ths = iv
            overrides['untracked_ths'] = iv
        iv = int(getattr(args, 'event_line_shadow_line_min_valid_step_count', -1) or -1)
        if iv > 0:
            shadow_args.line_min_valid_step_count = iv
            overrides['line_min_valid_step_count'] = iv
        try:
            return build_line_filter(shadow_args), {
                'enabled': True,
                'camera': camera_short_alias,
                'overrides': overrides,
                'reason': 'active',
            }
        except Exception as exc:
            return None, {
                'enabled': False,
                'camera': camera_short_alias,
                'overrides': overrides,
                'reason': f'init_failed:{type(exc).__name__}:{exc}',
            }

    shadow_line_filter, shadow_line_filter_config = _build_shadow_line_filter()
    if hot_pixel_filter.enabled:
        print(
            f"[HotPixel][{camera_log_tag}] enabled mode={hot_pixel_filter.mode} "
            f"radius={hot_pixel_filter.radius_px} static={sorted(hot_pixel_filter.static_pixels)} "
            f"learn={hot_pixel_filter.learn_enable}",
            flush=True,
        )
    if shadow_line_filter_config.get('enabled'):
        print(f"[LineFilterShadow][{camera_log_tag}] {shadow_line_filter_config}", flush=True)

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
        nonlocal spatter_tracker, clusters, line_filter, shadow_line_filter, shadow_line_filter_config
        spatter_tracker = SpatterTrackerAlgorithm(**spatter_tracker_kwargs)
        clusters = EventSpatterClusterBuffer()
        for roi in input_nozone_rois:
            center_x, center_y, radius, inside = roi
            spatter_tracker.add_nozone(np.array([center_x, center_y]), radius, inside)
        line_filter = build_line_filter(args)
        shadow_line_filter, shadow_line_filter_config = _build_shadow_line_filter()
        print(f'[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] reset detection state: {reason}', flush=True)

    if args.disable_spatter_processing:
        print(
            f'[SpatterDebug][{camera_log_tag}][SN={camera_sn_for_log}] spatter processing disabled; RAW/TSF1/render path only.',
            flush=True)
    if cluster_guard_enabled:
        print(
            f'[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] cluster guard enabled: thresh={cluster_guard_thresh}',
            flush=True)
    if pre_spatter_guard_enabled:
        print(
            f'[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] pre-spatter guard enabled: '
            f'max_events={pre_spatter_raw_thresh}, rate_thresh={pre_spatter_rate_thresh:.2f} MEv/s, '
            f'trigger_slices={pre_spatter_trigger_slices}, hold_ms={int(pre_spatter_hold_us / 1000)}',
            flush=True,
        )

    hal_raw_recorder = HotkeyRawRecorder(
        device=hal_device,
        enabled=args.enable_hotkey_raw_record and live and hal_device is not None,
        output_dir=args.raw_record_dir,
        prefix=args.raw_record_prefix,
        camera_id=camera_short_alias,
    )
    event_slice_recorder = EventSlicePackRecorder(
        enabled=bool(getattr(args, 'event_slice_record_enable', True)) and live,
        output_root=str(getattr(args, 'event_slice_record_root', './recordings') or './recordings'),
        camera_id=camera_short_alias,
        serial=str(getattr(args, 'tsf1_camera_id', '') or ''),
        width=width,
        height=height,
        queue_limit=int(getattr(args, 'event_slice_record_queue', 512) or 512),
        flush_interval_ms=int(getattr(args, 'event_slice_record_flush_ms', 1000) or 1000),
    )
    raw_recorder = CombinedEventRawRecorder(
        [hal_raw_recorder, event_slice_recorder],
        camera_id=camera_short_alias,
    )
    if hal_raw_recorder.enabled and hal_raw_recorder.can_record():
        print(f'[RAW] hotkey recording enabled. Press R to start/stop. dir={args.raw_record_dir}', flush=True)
    elif args.enable_hotkey_raw_record and live:
        print('[RAW] Metavision HAL raw logging unavailable on this path; using event-slice replay-pack only.',
              flush=True)
    if bool(getattr(args, 'event_slice_record_enable', True)) and live:
        print(
            f'[RAW][EVP] event-slice replay-pack recording enabled. '
            f'camera={camera_short_alias} root={getattr(args, "event_slice_record_root", "./recordings")}',
            flush=True,
        )

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

    do_filtering = args.stc_threshold > 0
    filtered_evs = SpatioTemporalContrastAlgorithm.get_empty_output_buffer() if do_filtering else None
    if do_filtering:
        stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_threshold, False)

    event_human_mask_filter = CurrentFrameHumanMaskEventFilter(
        args=args,
        width=width,
        height=height,
        camera_alias=camera_short_alias,
    )
    event_human_mask_buffer = HumanMaskSyncEventBuffer(args=args, camera_alias=camera_short_alias)
    track_polygon_mask_provider = None
    track_polygon_gate = None
    track_polygon_mode = str(getattr(args, 'event_track_polygon_gate_mode', 'off') or 'off').strip().lower()
    track_polygon_cameras = {
        item.strip().upper()
        for item in re.split(r'[,;\s]+', str(getattr(args, 'event_track_polygon_gate_cameras', '') or ''))
        if item.strip()
    }
    track_polygon_selected = (
        not track_polygon_cameras
        or str(camera_short_alias or '').strip().upper() in track_polygon_cameras
        or 'ALL' in track_polygon_cameras
        or '*' in track_polygon_cameras
    )
    if track_polygon_mode in ('shadow', 'active') and track_polygon_selected:
        # The provider only reads same-sync polygons. It never enters the event
        # prefilter path, so raw tracking input remains intact in shadow mode.
        provider_args = copy.copy(args)
        provider_args.event_human_mask_filter_enable = True
        provider_args.event_human_mask_mode = 'visual_only'
        provider_args.event_human_mask_current_only = True
        provider_args.event_human_mask_wait_ms = 0.0
        provider_args.event_human_mask_use_carry_last = False
        provider_args.event_human_mask_dilate_px = max(
            0, int(getattr(args, 'event_track_polygon_gate_dilate_px', 0) or 0)
        )
        track_polygon_mask_provider = CurrentFrameHumanMaskEventFilter(
            args=provider_args,
            width=width,
            height=height,
            camera_alias=camera_short_alias,
        )
        track_polygon_gate = EventTrackPolygonGate(
            args=args,
            width=width,
            height=height,
            camera_alias=camera_short_alias,
            mask_provider=track_polygon_mask_provider,
        )
        print(
            f"[V26B][{camera_log_tag}] polygon_gate mode={track_polygon_mode} "
            f"enabled={track_polygon_gate.enabled} hold_ms={track_polygon_gate.hold_us / 1000.0:.1f}",
            flush=True,
        )
    business_id_v1_shadow = BusinessIdV1Shadow(args, camera_short_alias)
    business_id_v3_shadow = BusinessIdV3Shadow(args, camera_short_alias)
    event_sync_mapper = MvsSoftTriggerTimelineMapper(
        args=args,
        camera_alias=camera_short_alias,
        camera_serial=args.tsf1_camera_id,
    )
    event_replay_mvs_pacer = EventReplayMvsPacer(args)

    input_nozone_rois = args.nozone
    for roi in input_nozone_rois:
        center_x, center_y, radius, inside = roi
        spatter_tracker.add_nozone(np.array([center_x, center_y]), radius, inside)
        print(f'No-zone: center=({center_x},{center_y}) radius={radius} inside={inside}')

    events_frame_gen_algo = OnDemandFrameGenerationAlgorithm(width, height, args.accumulation_time_us)
    output_img = np.zeros((height, width, 3), np.uint8)
    log = []

    debug_csv_file = None
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
        'last_mapped_mvs_sync_index': None,
        'last_mapped_mvs_sync_event_ts_us': None,
        'last_event_sync_map': {},
        'sync_start_tick_id': 0,
        # PATCH ext-trigger state init begin
        'last_accepted_selected_event_ts_us': None,
        'dedupe_skipped_edges': 0,
        # PATCH ext-trigger state init end
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
            'mapped_mvs_sync_index', 'sync_master', 'event_local_sync_index',
            'mvs_soft_trigger_event_ts_us',
            'mvs_trigger_to_event_edge_us', 'mvs_trigger_to_event_edge_us_used',
            'mvs_soft_trigger_event_ts_source',
            'sync_lock_state', 'sync_match_mode',
        ])
        ext_trigger_writer.writeheader()
        print(
            f"[SYNC] external trigger log enabled -> {ext_trigger_path}; "
            f"selected_polarity={int(getattr(args, 'ext_trigger_polarity', 0))} "
            f"log_all_edges={bool(getattr(args, 'ext_trigger_log_all_edges', False))}",
            flush=True,
        )
        if ext_trigger_gate_required:
            print(
                f"[SYNC] formal sync_start gate enabled: start_file={ext_trigger_sync_start_path}; "
                f"pre-start trigger edges will be discarded and sync_index will reset to 0",
                flush=True,
            )

    tsf1_pub_writer = None
    tsf1_sender_proc = None
    if args.enable_tsf1_pub:
        pub_scale = max(0.05, float(args.tsf1_pub_scale))
        pub_w = max(1, int(round(width * pub_scale)))
        pub_h = max(1, int(round(height * pub_scale)))
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
        overlay_channels = 1 if int(getattr(args, 'event_overlay_pub_channels', 3) or 3) == 1 else 3
        event_overlay_pub_writer = SharedFrameWriter(
            str(getattr(args, 'event_overlay_pub_name', 'event_overlay_latest') or 'event_overlay_latest'),
            overlay_w, overlay_h, channels=overlay_channels, create=True
        )
        print(
            f"[EventOverlayPub] shared memory enabled: name={args.event_overlay_pub_name}, "
            f"size={overlay_w}x{overlay_h}, channels={overlay_channels}, "
            f"mode={args.event_overlay_pub_mode}, render={getattr(args, 'event_overlay_render_mode', 'bgr')}",
            flush=True,
        )

    tsf1_pub_limiter = _SimpleRateLimiter(float(getattr(args, 'event_tsf1_pub_max_fps', 170.0) or 0.0), 'tsf1')
    event_overlay_pub_limiter = _SimpleRateLimiter(float(getattr(args, 'event_overlay_pub_max_fps', 170.0) or 0.0), 'event_overlay')
    obs_submit_limiter = _SimpleRateLimiter(float(getattr(args, 'event_obs_submit_max_fps', 170.0) or 0.0), 'obs')
    event_viz_build_limiter = _SimpleRateLimiter(float(getattr(args, 'event_viz_build_max_fps', 0.0) or 0.0), 'event_viz_build')
    event_viz_empty_heartbeat_limiter = _SimpleRateLimiter(
        float(getattr(args, 'event_viz_idle_heartbeat_fps', 20.0) or 0.0),
        'event_viz_empty_heartbeat',
    )
    event_viz_skip_empty_fastpath = bool(getattr(args, 'event_viz_skip_empty_fastpath', False))
    event_viz_mainloop_stats = {
        'skip_empty_fastpath': bool(event_viz_skip_empty_fastpath),
        'empty_fastpath_seen': 0,
        'empty_fastpath_skipped': 0,
        'empty_fastpath_heartbeat': 0,
        'empty_fastpath_pushed': 0,
        'buffer_wait_heartbeat_pushed': 0,
        'buffer_release_viz_only_pushed': 0,
        'build_rate_skipped': 0,
        'build_none': 0,
        'push_failed': 0,
    }

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
                max_consecutive_failures=int(getattr(args, 'obs_restart_max_consecutive_failures', 8)),
                circuit_breaker_sec=float(getattr(args, 'obs_restart_circuit_breaker_sec', 60.0)),
                max_backoff_sec=float(getattr(args, 'obs_restart_max_backoff_sec', 30.0)),
            )
            # OBS bullet persistence compatibility: only affects OBS output, not local detection/window.
            obs_streamer.bullet_persist_enabled = bool(getattr(args, 'obs_bullet_persist', False))
            obs_streamer.bullet_persist_decay = float(getattr(args, 'obs_bullet_persist_decay', 0.90))
            obs_streamer.bullet_persist_modes = str(getattr(args, 'obs_bullet_persist_modes', 'bullet') or 'bullet')
            obs_streamer.bullet_persist_alpha = float(getattr(args, 'obs_bullet_persist_alpha', 1.0))
            obs_streamer.bullet_persist_tau_ms = float(getattr(args, 'obs_bullet_persist_tau_ms', 100.0))
            obs_streamer.start()

    event_status_writer = EventWorkerStatusWriter(args, camera_short_alias, width, height)
    event_loop_profiler = EventLoopStageProfiler(args, camera_short_alias)
    event_rate_limiters = {
        'tsf1': tsf1_pub_limiter,
        'event_overlay': event_overlay_pub_limiter,
        'obs': obs_submit_limiter,
        'event_viz_build': event_viz_build_limiter,
        'event_viz_empty_heartbeat': event_viz_empty_heartbeat_limiter,
    }

    event_viz_buffer = None
    event_viz_publisher = None
    if bool(getattr(args, 'event_viz_buffer_enable', True)) and str(getattr(args, 'event_viz_worker_mode', 'thread') or 'thread').lower() != 'off':
        try:
            max_fps_for_slots = max(
                float(getattr(args, 'event_overlay_pub_max_fps', 170.0) or 0.0),
                float(getattr(args, 'event_obs_submit_max_fps', 170.0) or 0.0),
                float(getattr(args, 'event_tsf1_pub_max_fps', 170.0) or 0.0),
                float(getattr(args, 'event_viz_idle_heartbeat_fps', 20.0) or 0.0),
            )
            if int(getattr(args, 'event_viz_buffer_slots', 0) or 0) > 0:
                viz_slots = int(getattr(args, 'event_viz_buffer_slots'))
            else:
                buf_ms = max(1.0, float(getattr(args, 'event_viz_buffer_ms', 50.0) or 50.0))
                slice_ms = max(0.1, float(getattr(args, 'accumulation_time_us', 4000) or 4000) / 1000.0)
                pub_ms = 1000.0 / max(1.0, max_fps_for_slots) if max_fps_for_slots > 0 else slice_ms
                viz_slots = _next_power_of_two(max(4, int(np.ceil(buf_ms / slice_ms)), int(np.ceil(buf_ms / pub_ms))))
            event_viz_buffer = EventVizBuffer(
                buffer_ms=float(getattr(args, 'event_viz_buffer_ms', 50.0) or 50.0),
                slots=viz_slots,
                drop_policy=str(getattr(args, 'event_viz_buffer_drop_policy', 'drop_oldest') or 'drop_oldest'),
            )
            event_viz_publisher = EventVizPublisher(
                args=args, camera_alias=camera_short_alias, width=width, height=height,
                buffer=event_viz_buffer,
                event_overlay_pub_writer=event_overlay_pub_writer,
                obs_streamer=obs_streamer,
                tsf1_publisher=tsf1_publisher,
                rate_limiters=event_rate_limiters,
            )
            event_viz_publisher.start()
            print(
                f"[EventViz][{camera_log_tag}] enabled source=post_human_mask buffer_ms={event_viz_buffer.buffer_ms:.1f} "
                f"slots={event_viz_buffer.slots} overlay={event_overlay_pub_writer is not None} "
                f"obs={obs_streamer is not None} tsf1={tsf1_publisher is not None}",
                flush=True,
            )
        except Exception as exc:
            print(f"[EventViz][WARN][{camera_log_tag}] init failed: {type(exc).__name__}: {exc}", flush=True)
            event_viz_buffer = None
            event_viz_publisher = None

    def _event_viz_enabled():
        return event_viz_buffer is not None and event_viz_publisher is not None

    def _maybe_publish_tsf1(buffer, ts_i, allow_guard=True):
        if _event_viz_enabled():
            return False
        if tsf1_publisher is None or not allow_guard:
            return False
        if not tsf1_pub_limiter.allow():
            return False
        tsf1_publisher.update_events(buffer)
        tsf1_publisher.publish_up_to(ts_i)
        return True

    def _limited_overlay_writer():
        if _event_viz_enabled():
            return None
        if event_overlay_pub_writer is None:
            return None
        return event_overlay_pub_writer if event_overlay_pub_limiter.allow() else None

    def _limited_obs_submit(enabled=True):
        if _event_viz_enabled():
            return False
        return bool(enabled) and obs_submit_limiter.allow()

    def _build_event_viz_packet(raw_evs, filtered_evs, ts_i, wall_time_us_i, sync_index_i, sync_event_ts_i,
                                fastpath_reason='', track_evs=None, track_input_policy='post_mask'):
        if not _event_viz_enabled():
            return None
        try:
            raw_n_i = int(len(raw_evs)) if raw_evs is not None else 0
        except Exception:
            raw_n_i = 0
        try:
            filt_n_i = int(len(filtered_evs)) if filtered_evs is not None else 0
        except Exception:
            filt_n_i = 0
        try:
            track_n_i = int(len(track_evs)) if track_evs is not None else filt_n_i
        except Exception:
            track_n_i = filt_n_i
        stats = event_human_mask_filter.get_last_stats()
        mask_valid = bool(stats.get('mask_pixels', 0) or stats.get('used_persons', 0))
        no_mask_policy = str(getattr(args, 'event_viz_no_mask_policy', 'empty') or 'empty').lower()
        viz_after_mask = bool(getattr(args, 'event_viz_after_human_mask_filter', True))
        filter_policy = 'post_human_mask'
        if viz_after_mask:
            evs_for_viz = filtered_evs
            if (not mask_valid) and no_mask_policy == 'skip':
                return None
            if (not mask_valid) and no_mask_policy == 'empty':
                try:
                    evs_for_viz = raw_evs[:0].copy() if raw_evs is not None else filtered_evs
                except Exception:
                    evs_for_viz = filtered_evs
            elif (not mask_valid) and no_mask_policy == 'filtered':
                evs_for_viz = filtered_evs
            elif (not mask_valid) and no_mask_policy == 'raw_debug':
                evs_for_viz = raw_evs
                filter_policy = 'raw_debug_no_mask'
        else:
            evs_for_viz = raw_evs
            filter_policy = 'raw_pre_human_mask'
        evs_copy = _copy_events_for_viz(evs_for_viz)
        try:
            viz_n_i = int(len(evs_copy)) if evs_copy is not None else 0
        except Exception:
            viz_n_i = 0
        t_min, t_max, t_mid = _event_ts_range(evs_copy if evs_copy is not None and len(evs_copy) > 0 else raw_evs, int(ts_i or 0))
        human_capture_us = _safe_int(stats.get('human_capture_wall_us'), 0)
        human_record_us = _safe_int(stats.get('human_record_wall_us'), 0)
        now_wall_us = int(wall_time_us_i or _now_wall_time_us())
        human_age_ms = ((now_wall_us - human_record_us) / 1000.0) if human_record_us > 0 else -1.0
        event_to_human_ms = ((now_wall_us - human_capture_us) / 1000.0) if human_capture_us > 0 else float('nan')
        reduction = 0.0 if raw_n_i <= 0 else max(0.0, min(1.0, 1.0 - float(filt_n_i) / float(max(1, raw_n_i))))
        pkt = EventVizPacket(
            camera=camera_short_alias,
            seq=int(getattr(_build_event_viz_packet, '_seq', 0)) + 1,
            source_stage='post_human_mask',
            events=evs_copy,
            event_ts_min_us=t_min,
            event_ts_max_us=t_max,
            event_ts_mid_us=t_mid,
            event_wall_us=now_wall_us,
            event_mono_ns=int(time.monotonic_ns()),
            trigger_seq=_safe_int(sync_index_i, -1),
            program_sync_index=_safe_int(sync_index_i, -1),
            mvs_frame_id=_safe_int(stats.get('sync_index'), _safe_int(sync_index_i, -1)),
            human_frame_id=_safe_int(stats.get('human_frame_id'), _safe_int(stats.get('sync_index'), -1)),
            human_mvs_frame_id=_safe_int(stats.get('human_mvs_frame_id'), _safe_int(stats.get('sync_index'), -1)),
            human_capture_wall_us=int(human_capture_us),
            human_record_wall_us=int(human_record_us),
            human_result_age_ms=float(human_age_ms),
            event_to_human_capture_delta_ms=float(event_to_human_ms),
            event_to_mvs_delta_ms=float('nan'),
            raw_event_count=int(raw_n_i),
            filtered_event_count=int(filt_n_i),
            track_event_count=int(track_n_i),
            viz_event_count=int(viz_n_i),
            mask_valid=bool(mask_valid),
            mask_source=str(stats.get('infer_source', '') or stats.get('reason', '') or ''),
            mask_person_count=_safe_int(stats.get('used_persons'), _safe_int(stats.get('person_count'), 0)),
            mask_age_ms=float(human_age_ms),
            mask_stale=bool(human_age_ms > float(getattr(args, 'event_human_mask_cache_stale_ms', 150.0) or 150.0)) if human_age_ms >= 0 else True,
            filter_policy=str(filter_policy),
            track_input_policy=str(track_input_policy or ''),
            filter_reduction_ratio=float(reduction),
            empty_after_filter=bool(filt_n_i <= 0),
            empty_after_viz=bool(viz_n_i <= 0),
            fastpath_reason=str(fastpath_reason or ''),
            push_wall_us=int(_now_wall_time_us()),
            push_mono_ns=int(time.monotonic_ns()),
        )
        setattr(_build_event_viz_packet, '_seq', int(pkt.seq))
        return pkt

    def _push_event_viz_packet(raw_evs, filtered_evs, ts_i, wall_time_us_i, sync_index_i, sync_event_ts_i,
                               fastpath_reason='', track_evs=None, track_input_policy='post_mask'):
        if not _event_viz_enabled():
            return False
        reason = str(fastpath_reason or '')
        is_empty_fastpath = (
            reason in ('empty_slice', 'mask_filtered_empty', 'track_input_empty')
            or reason.startswith('human_mask_buffer')
            or reason.startswith('waiting_')
            or reason.endswith('_drop')
        )
        if is_empty_fastpath:
            event_viz_mainloop_stats['empty_fastpath_seen'] += 1
            if event_viz_skip_empty_fastpath:
                if event_viz_empty_heartbeat_limiter is not None and not event_viz_empty_heartbeat_limiter.allow():
                    event_viz_mainloop_stats['empty_fastpath_skipped'] += 1
                    return False
                event_viz_mainloop_stats['empty_fastpath_heartbeat'] += 1
        if event_viz_build_limiter is not None and not event_viz_build_limiter.allow():
            event_viz_mainloop_stats['build_rate_skipped'] += 1
            return False
        pkt = _build_event_viz_packet(
            raw_evs, filtered_evs, ts_i, wall_time_us_i, sync_index_i, sync_event_ts_i,
            fastpath_reason, track_evs=track_evs, track_input_policy=track_input_policy,
        )
        if pkt is None:
            event_viz_mainloop_stats['build_none'] += 1
            return False
        pushed = bool(event_viz_buffer.try_push(pkt))
        if pushed and is_empty_fastpath:
            event_viz_mainloop_stats['empty_fastpath_pushed'] += 1
        elif not pushed:
            event_viz_mainloop_stats['push_failed'] += 1
        return pushed

    def _empty_events_like(evs):
        if evs is None:
            return None
        try:
            return evs[:0].copy()
        except Exception:
            return None

    def _select_track_input(raw_evs_i, detection_evs_i):
        policy = str(getattr(args, 'event_track_input_policy', 'post_mask') or 'post_mask').strip().lower()
        stats_i = event_human_mask_filter.get_last_stats()
        mask_available = bool(
            stats_i.get('mask_available')
            or int(stats_i.get('mask_pixels', 0) or 0) > 0
            or int(stats_i.get('used_persons', 0) or 0) > 0
        )
        try:
            raw_count = int(len(raw_evs_i)) if raw_evs_i is not None else 0
        except Exception:
            raw_count = 0
        try:
            det_count = int(len(detection_evs_i)) if detection_evs_i is not None else 0
        except Exception:
            det_count = 0
        use_raw = False
        reason = 'post_mask'
        if policy == 'raw':
            use_raw = True
            reason = 'raw'
        elif policy == 'raw_on_mask_unavailable' and not mask_available:
            use_raw = True
            reason = 'raw_on_mask_unavailable'
        elif policy == 'raw_when_filtered_empty' and raw_count > 0 and det_count <= 0:
            use_raw = True
            reason = 'raw_when_filtered_empty'
        elif policy == 'raw_on_mask_unavailable_or_filtered_empty' and (not mask_available or (raw_count > 0 and det_count <= 0)):
            use_raw = True
            reason = 'raw_on_mask_unavailable_or_filtered_empty'
        track_evs_i = raw_evs_i if use_raw else detection_evs_i
        try:
            track_count = int(len(track_evs_i)) if track_evs_i is not None else 0
        except Exception:
            track_count = 0
        return track_evs_i, int(track_count), str(reason if use_raw else 'post_mask'), bool(use_raw)

    def _refresh_bullet_display_diag(debug_rows=None):
        nonlocal bullet_display_diag
        diag = dict(bullet_display_diag)
        if debug_rows is not None:
            active_count = 0
            display_eligible_count = 0
            try:
                for row in (debug_rows or []):
                    if not isinstance(row, dict):
                        continue
                    active = bool(_safe_int(row.get('bullet_active', row.get('accepted_now', 0)), 0))
                    accepted = bool(_safe_int(row.get('accepted_now', 0), 0))
                    if active or accepted:
                        active_count += 1
                    if _safe_int(row.get('display_bullet_id', -1), -1) >= 0:
                        display_eligible_count += 1
            except Exception:
                active_count = int(diag.get('active_track_count', 0) or 0)
                display_eligible_count = int(diag.get('display_eligible_count', 0) or 0)
            diag['active_track_count'] = int(active_count)
            diag['display_eligible_count'] = int(display_eligible_count)
            diag['active_but_not_display_count'] = int(max(0, active_count - display_eligible_count))
        try:
            extractor_stats = bullet_extractor.stats() if hasattr(bullet_extractor, 'stats') else {}
        except Exception:
            extractor_stats = {}
        try:
            sender_stats = bullet_sender.stats() if hasattr(bullet_sender, 'stats') else {}
        except Exception:
            sender_stats = {}
        diag['draw_paths_count'] = int(_safe_int(extractor_stats.get('draw_paths_count', diag.get('draw_paths_count', 0)), 0))
        diag['draw_path_points_count'] = int(_safe_int(extractor_stats.get('draw_path_points_count', diag.get('draw_path_points_count', 0)), 0))
        diag['last_overlay_bullet_point_age_ms'] = float(_safe_float(extractor_stats.get('last_overlay_bullet_point_age_ms', diag.get('last_overlay_bullet_point_age_ms', -1.0)), -1.0))
        diag['last_overlay_bullet_point_ts_us'] = int(_safe_int(extractor_stats.get('last_overlay_bullet_point_ts_us', diag.get('last_overlay_bullet_point_ts_us', 0)), 0))
        diag['visual_point_submit_count'] = int(_safe_int(extractor_stats.get('visual_point_submit_count', diag.get('visual_point_submit_count', 0)), 0))
        diag['visual_point_submit_failed_count'] = int(_safe_int(extractor_stats.get('visual_point_submit_failed_count', diag.get('visual_point_submit_failed_count', 0)), 0))
        diag['visual_point_skip_count'] = int(_safe_int(extractor_stats.get('visual_point_skip_count', diag.get('visual_point_skip_count', 0)), 0))
        prev_dropped = int(diag.get('bullet_sender_dropped', 0) or 0)
        diag['bullet_sender_queue_size'] = int(_safe_int(sender_stats.get('queue_size', diag.get('bullet_sender_queue_size', 0)), 0))
        diag['bullet_sender_queue_max_size'] = int(_safe_int(sender_stats.get('queue_max_size', diag.get('bullet_sender_queue_max_size', 0)), 0))
        diag['bullet_sender_sent'] = int(_safe_int(sender_stats.get('sent_count', diag.get('bullet_sender_sent', 0)), 0))
        diag['bullet_sender_dropped'] = int(_safe_int(sender_stats.get('drop_count', diag.get('bullet_sender_dropped', 0)), 0))
        diag['bullet_sender_drop_delta'] = int(max(0, int(diag.get('bullet_sender_dropped', 0) or 0) - prev_dropped))
        diag['bullet_sender_drop_by_type'] = sender_stats.get('drop_by_type', diag.get('bullet_sender_drop_by_type', {}))
        diag['bullet_sender_camera_alias'] = str(sender_stats.get('camera_alias', diag.get('bullet_sender_camera_alias', '')) or '')
        diag['bullet_sender_name'] = str(sender_stats.get('name', diag.get('bullet_sender_name', '')) or '')
        age_ms = float(diag.get('last_overlay_bullet_point_age_ms', -1.0) or -1.0)
        active_count = int(diag.get('active_track_count', 0) or 0)
        display_count = int(diag.get('display_eligible_count', 0) or 0)
        draw_count = int(diag.get('draw_paths_count', 0) or 0)
        queue_size = int(diag.get('bullet_sender_queue_size', 0) or 0)
        queue_max = int(diag.get('bullet_sender_queue_max_size', 0) or 0)
        drop_delta = int(diag.get('bullet_sender_drop_delta', 0) or 0)
        if drop_delta > 0 or (queue_max > 0 and queue_size >= max(1, int(queue_max * 0.75))):
            source_state = 'sender_backpressure'
        elif age_ms >= 0.0 and age_ms <= 500.0:
            source_state = 'ok'
        elif active_count > 0 and display_count <= 0:
            source_state = 'active_track_not_displayed'
        elif draw_count <= 0 and active_count <= 0:
            source_state = 'draw_paths_empty_no_active_track'
        elif draw_count > 0 and (age_ms < 0.0 or age_ms > 500.0):
            source_state = 'draw_paths_present_sender_stale'
        else:
            source_state = 'idle'
        diag['bullet_source_state'] = source_state
        bullet_display_diag = diag
        return dict(diag)

    def _update_event_worker_status(state='running', active_reason='', ts_i=0, raw_n_i=0, detection_n_i=0,
                                    raw_rate_mevs_i=0.0, fastpath=False, cluster_count_i=0, accepted_count_i=0,
                                    guard_state='OK', force=False, track_n_i=None, track_input_policy_i='',
                                    track_path_position_i='', track_raw_n_i=None,
                                    mask_released_raw_n_i=0, mask_filtered_n_i=0,
                                    mask_release_consumed_for_viz_only_i=False,
                                    mask_buffer_blocks_tracking_i=None,
                                    track_input_starved_reason_i='',
                                    raw_evs_i=None, track_evs_i=None,
                                    clusters_np_i=None, accepted_clusters_i=None,
                                    debug_rows_i=None,
                                    hot_pixel_filter_stats_i=None,
                                    cluster_sanitize_diag_i=None,
                                    line_filter_shadow_diag_i=None,
                                    track_polygon_gate_diag_i=None,
                                    business_id_v1_shadow_diag_i=None,
                                    business_id_v3_shadow_diag_i=None,
                                    track_n_before_hot_pixel_i=None):
        try:
            event_status_writer.account_slice(raw_n=raw_n_i, fastpath=fastpath)
            if not event_status_writer.should_update(force=force):
                return
            bullet_diag = _refresh_bullet_display_diag()
            track_policy_s = str(track_input_policy_i or getattr(args, 'event_track_input_policy', 'post_mask') or 'post_mask')
            track_path_position_s = str(track_path_position_i or '')
            if not track_path_position_s:
                track_path_position_s = 'before_mask_buffer' if track_policy_s == 'raw_before_mask_buffer' else 'after_mask_buffer'
            if mask_buffer_blocks_tracking_i is None:
                mask_buffer_blocks_tracking_i = bool(
                    event_human_mask_buffer.enabled and track_path_position_s != 'before_mask_buffer'
                    and (str(active_reason or '').startswith('human_mask_buffer') or str(state or '') == 'waiting')
                )
            track_decoupled = bool(track_path_position_s == 'before_mask_buffer')
            fg_state = {}
            try:
                fg_state_all = getattr(_maybe_render, '_frame_gen_state', {})
                if isinstance(fg_state_all, dict):
                    fg_state = fg_state_all.get(int(id(events_frame_gen_algo)), {}) or {}
            except Exception:
                fg_state = {}
            raw_spatial_diag = _event_spatial_diag(raw_evs_i, width, height)
            track_spatial_diag = _event_spatial_diag(track_evs_i, width, height)
            cluster_spatial_diag = _cluster_spatial_diag(clusters_np_i, accepted_clusters_i, width, height)
            line_reject_diag = _line_filter_reject_diag(debug_rows_i)
            hot_pixel_diag = hot_pixel_filter_stats_i if isinstance(hot_pixel_filter_stats_i, dict) else {}
            cluster_sanitize_diag = cluster_sanitize_diag_i if isinstance(cluster_sanitize_diag_i, dict) else {}
            line_shadow_diag = line_filter_shadow_diag_i if isinstance(line_filter_shadow_diag_i, dict) else {}
            polygon_gate_diag = track_polygon_gate_diag_i if isinstance(track_polygon_gate_diag_i, dict) else {}
            business_v1_shadow_diag = business_id_v1_shadow_diag_i if isinstance(business_id_v1_shadow_diag_i, dict) else {}
            business_v3_shadow_diag = business_id_v3_shadow_diag_i if isinstance(business_id_v3_shadow_diag_i, dict) else {}
            track_event_count_for_diag = int(track_n_i if track_n_i is not None else detection_n_i or 0)
            if track_n_before_hot_pixel_i is not None:
                track_event_count_before_hot_pixel = int(track_n_before_hot_pixel_i)
            elif isinstance(hot_pixel_diag, dict):
                track_event_count_before_hot_pixel = int(hot_pixel_diag.get('input_count', track_event_count_for_diag) or 0)
            else:
                track_event_count_before_hot_pixel = int(track_event_count_for_diag)
            track_unique_pixels = int(track_spatial_diag.get('unique_pixels', 0) or 0)
            event_has_spatial_data = bool(track_event_count_for_diag >= 8 and track_unique_pixels > 0)
            draw_count_for_diag = int(bullet_diag.get('draw_paths_count', 0) or 0)
            active_count_for_diag = int(bullet_diag.get('active_track_count', 0) or 0)
            display_count_for_diag = int(bullet_diag.get('display_eligible_count', 0) or 0)
            cluster_count_for_diag = int(cluster_count_i or 0)
            accepted_count_for_diag = int(accepted_count_i or 0)
            if (
                track_event_count_for_diag <= 0
                and bool(hot_pixel_diag.get('enabled', False))
                and int(hot_pixel_diag.get('input_count', 0) or 0) > 0
                and int(hot_pixel_diag.get('output_count', 0) or 0) <= 0
            ):
                block_stage_reason = 'hot_pixel_filter_removed_all'
            elif track_event_count_for_diag <= 0:
                block_stage_reason = 'track_input_empty'
            elif bool(track_spatial_diag.get('spatial_collapse_suspect', False)):
                block_stage_reason = 'event_spatial_collapse_suspect'
            elif cluster_count_for_diag <= 0:
                block_stage_reason = 'spatter_no_cluster'
            elif accepted_count_for_diag <= 0:
                block_stage_reason = 'line_filter_rejected_all_clusters'
            elif active_count_for_diag <= 0:
                block_stage_reason = 'line_filter_no_active_track'
            elif display_count_for_diag <= 0:
                block_stage_reason = 'active_track_not_displayed'
            elif draw_count_for_diag <= 0:
                block_stage_reason = 'display_track_without_draw_path'
            else:
                block_stage_reason = 'ok'
            event_has_data_but_no_display_track = bool(
                event_has_spatial_data and draw_count_for_diag <= 0 and active_count_for_diag <= 0
            )
            event_status_writer.update(
                state=state, active_reason=active_reason, ts_us=int(ts_i or 0),
                raw_n=int(raw_n_i or 0), detection_n=int(detection_n_i or 0),
                raw_rate_mevs=float(raw_rate_mevs_i or 0.0),
                accumulation_time_us=int(getattr(args, 'accumulation_time_us', 0) or 0),
                fastpath=bool(fastpath), cluster_count=int(cluster_count_i or 0),
                accepted_count=int(accepted_count_i or 0), guard_state=str(guard_state or 'OK'),
                human_mask_stats=event_human_mask_filter.get_last_stats(),
                rate_limiters=event_rate_limiters,
                extra={
                    'track_event_count': int(track_n_i if track_n_i is not None else detection_n_i or 0),
                    'track_event_count_before_hot_pixel': int(track_event_count_before_hot_pixel),
                    'track_raw_event_count': int(track_raw_n_i if track_raw_n_i is not None else raw_n_i or 0),
                    'track_input_policy': track_policy_s,
                    'track_input_policy_effective': track_policy_s,
                    'track_path_position': track_path_position_s,
                    'track_decoupled_from_mask': bool(track_decoupled),
                    'mask_buffer_blocks_tracking': bool(mask_buffer_blocks_tracking_i),
                    'mask_released_event_count': int(mask_released_raw_n_i or 0),
                    'mask_filtered_event_count': int(mask_filtered_n_i or 0),
                    'mask_release_consumed_for_viz_only': bool(mask_release_consumed_for_viz_only_i),
                    'track_input_starved_reason': str(track_input_starved_reason_i or ''),
                    'raw_event_spatial_diag': raw_spatial_diag,
                    'track_event_spatial_diag': track_spatial_diag,
                    'spatter_cluster_diag': cluster_spatial_diag,
                    'line_filter_reject_diag': line_reject_diag,
                    'event_hot_pixel_filter': dict(hot_pixel_diag),
                    'event_cluster_sanitize': dict(cluster_sanitize_diag),
                    'line_filter_shadow_diag': dict(line_shadow_diag),
                    'event_track_polygon_gate': dict(polygon_gate_diag),
                    'event_business_id_v1_shadow': dict(business_v1_shadow_diag),
                    'event_business_id_v3_shadow': dict(business_v3_shadow_diag),
                    'event_has_data_but_no_display_track': bool(event_has_data_but_no_display_track),
                    'overlay_has_data_but_no_display_track': bool(event_has_data_but_no_display_track),
                    'event_block_stage_reason': str(block_stage_reason),
                    'event_spatial_collapse_suspect': bool(
                        raw_spatial_diag.get('spatial_collapse_suspect', False)
                        or track_spatial_diag.get('spatial_collapse_suspect', False)
                    ),
                    'raw_recording': bool(raw_recorder.is_recording()),
                    'raw_recording_status': raw_recorder.status() if hasattr(raw_recorder, 'status') else {},
                    'tsf1_enabled': bool(tsf1_publisher is not None),
                    'event_overlay_enabled': bool(event_overlay_pub_writer is not None),
                    'obs_enabled': bool(obs_streamer is not None),
                    'obs': obs_streamer.stats() if obs_streamer is not None and hasattr(obs_streamer, 'stats') else {'enabled': bool(obs_streamer is not None)},
                    'viz': {
                        'buffer': event_viz_buffer.stats() if event_viz_buffer is not None else {'enable': False},
                        'publisher': event_viz_publisher.stats() if event_viz_publisher is not None else {'enable': False},
                        'mainloop': dict(event_viz_mainloop_stats),
                        'frame_gen_backward_reset_count': int(fg_state.get('backward_reset_count', 0) or 0),
                        'frame_gen_backward_retry_count': int(fg_state.get('backward_retry_count', 0) or 0),
                        'frame_gen_last_generate_ts_us': None if fg_state.get('last_generate_ts_us') is None
                        else int(fg_state.get('last_generate_ts_us', 0) or 0),
                    },
                    'human_mask_buffer': event_human_mask_buffer.stats(),
                    'event_sync': event_sync_mapper.health(),
                    'event_replay_mvs_pacing': event_replay_mvs_pacer.stats(),
                    'bullet_display_diag': dict(bullet_diag),
                    'draw_paths_count': int(bullet_diag.get('draw_paths_count', 0) or 0),
                    'draw_path_points_count': int(bullet_diag.get('draw_path_points_count', 0) or 0),
                    'display_eligible_count': int(bullet_diag.get('display_eligible_count', 0) or 0),
                    'active_track_count': int(bullet_diag.get('active_track_count', 0) or 0),
                    'active_but_not_display_count': int(bullet_diag.get('active_but_not_display_count', 0) or 0),
                    'last_overlay_bullet_point_age_ms': float(bullet_diag.get('last_overlay_bullet_point_age_ms', -1.0) or -1.0),
                    'bullet_sender_queue_size': int(bullet_diag.get('bullet_sender_queue_size', 0) or 0),
                    'bullet_sender_dropped': int(bullet_diag.get('bullet_sender_dropped', 0) or 0),
                    'bullet_sender_drop_delta': int(bullet_diag.get('bullet_sender_drop_delta', 0) or 0),
                    'bullet_sender_sent': int(bullet_diag.get('bullet_sender_sent', 0) or 0),
                    'bullet_sender_camera_alias': str(bullet_diag.get('bullet_sender_camera_alias', '') or ''),
                    'bullet_sender_name': str(bullet_diag.get('bullet_sender_name', '') or ''),
                    'visual_point_submit_count': int(bullet_diag.get('visual_point_submit_count', 0) or 0),
                    'visual_point_submit_failed_count': int(bullet_diag.get('visual_point_submit_failed_count', 0) or 0),
                    'visual_point_skip_count': int(bullet_diag.get('visual_point_skip_count', 0) or 0),
                    'bullet_source_state': str(bullet_diag.get('bullet_source_state', '')),
                    'ext_trigger': {
                        'all_edge_count': int(ext_trigger_state.get('all_index', 0) or 0),
                        'selected_edge_count': int(ext_trigger_state.get('sync_index', 0) or 0),
                        'last_selected_sync_index': ext_trigger_state.get('last_selected_sync_index'),
                        'last_mapped_mvs_sync_index': ext_trigger_state.get('last_mapped_mvs_sync_index'),
                        'dedupe_skipped_edges': int(ext_trigger_state.get('dedupe_skipped_edges', 0) or 0),
                        'nonselected_log_skipped_edges': int(ext_trigger_state.get('nonselected_log_skipped_edges', 0) or 0),
                        'log_all_edges': bool(getattr(args, 'ext_trigger_log_all_edges', False)),
                        'csv_selected_only': not bool(getattr(args, 'ext_trigger_log_all_edges', False)),
                        'selected_polarity': int(getattr(args, 'ext_trigger_polarity', 0) or 0),
                        'sync_start_tick_id': int(ext_trigger_state.get('sync_start_tick_id', 0) or 0),
                    },
                    'loop_profiler': event_loop_profiler.stats(),
                },
                force=force,
                account=False,
            )
        except Exception:
            pass

    window = None
    video_writer = None
    video_name = ''

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
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_name = _build_alias_output_path(args.out_video, camera_short_alias, args.tsf1_camera_id,
                                                  default_ext='.avi')
            video_fps = max(1.0, 1e6 / (args.accumulation_time_us * max(1, args.render_every_n)))
            video_writer = cv2.VideoWriter(video_name, fourcc, video_fps, (width, height))

        _write_ext_trigger_ready_file(args, camera_short_alias, args.tsf1_camera_id)

        iter_idx = 0
        for evs in mv_iterator:
            event_loop_profiler.start_slice()
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
                sync_mapper=event_sync_mapper,
            )
            ext_trigger_written = _drain_external_triggers(
                mv_iterator=mv_iterator,
                args=args,
                writer=ext_trigger_writer,
                state=ext_trigger_state,
                slice_ts_us=int(ts),
                wall_time_us=int(capture_wall_time_us),
                sync_mapper=event_sync_mapper,
            )
            if ext_trigger_written and ext_trigger_file is not None:
                # 同步链路需要 MVS 进程实时读取 event_trigger_*.csv。
                # 这里必须主动 flush，否则 trigger 行可能长时间停留在 Python 文件缓冲区，
                # MVS 端会误判为“no event_ts_us yet / offset 无法锁定”。
                ext_trigger_file.flush()
            if window is not None:
                EventLoop.poll_and_dispatch()
            event_loop_profiler.mark('trigger_sync_window')

            raw_evs = evs
            raw_n = int(len(raw_evs))
            current_event_local_sync_index_for_diag = ext_trigger_state.get('last_selected_sync_index')
            current_sync_map_for_diag = (
                ext_trigger_state.get('last_event_sync_map')
                if isinstance(ext_trigger_state.get('last_event_sync_map'), dict) else {}
            )
            current_sync_index_for_mask = ext_trigger_state.get('last_mapped_mvs_sync_index')
            current_sync_event_ts_for_mask = ext_trigger_state.get('last_mapped_mvs_sync_event_ts_us')
            if current_sync_index_for_mask is None:
                current_sync_index_for_mask = current_event_local_sync_index_for_diag
                current_sync_event_ts_for_mask = ext_trigger_state.get('last_selected_event_ts_us')
            event_replay_pacing_diag = event_replay_mvs_pacer.maybe_wait(
                current_sync_index_for_mask,
                direct_index_map=bool(ext_trigger_state.get('event_sync_direct_index_map', False)),
            )
            event_loop_profiler.mark('event_replay_mvs_pacing')
            sync_stats_extra = {
                'sync_master': str(current_sync_map_for_diag.get('sync_master', 'mvs_soft_trigger' if event_sync_mapper.enabled else 'event_local')),
                'event_local_sync_index': None if current_event_local_sync_index_for_diag is None else int(current_event_local_sync_index_for_diag),
                'mapped_mvs_sync_index': None if current_sync_map_for_diag.get('mapped_mvs_sync_index') is None else int(current_sync_map_for_diag.get('mapped_mvs_sync_index')),
                'event_sync_lock_state': str(current_sync_map_for_diag.get('lock_state', event_sync_mapper.lock_state)),
                'event_sync_match_mode': str(current_sync_map_for_diag.get('match_mode', '')),
                'mvs_trigger_to_event_edge_us': current_sync_map_for_diag.get('mvs_trigger_to_event_edge_us'),
                'mvs_trigger_to_event_edge_us_used': current_sync_map_for_diag.get('mvs_trigger_to_event_edge_us_used'),
                'mvs_soft_trigger_event_ts_us': current_sync_map_for_diag.get('mvs_soft_trigger_event_ts_us'),
                'mvs_soft_trigger_event_ts_source': current_sync_map_for_diag.get('mvs_soft_trigger_event_ts_source'),
                'rgb_exposure_as_sync_anchor': False,
                'event_replay_mvs_pacing': event_replay_pacing_diag,
            }
            raw_recorder.submit(
                raw_evs,
                int(ts),
                int(capture_wall_time_us),
                sync_meta={
                    'event_local_sync_index': current_event_local_sync_index_for_diag,
                    'mapped_mvs_sync_index': current_sync_map_for_diag.get('mapped_mvs_sync_index'),
                    'sync_event_ts_us': current_sync_event_ts_for_mask,
                },
            )
            event_loop_profiler.mark('raw_len_sync_lookup')
            requested_track_policy = str(getattr(args, 'event_track_input_policy', 'post_mask') or 'post_mask').strip().lower()
            track_before_mask_buffer = bool(event_human_mask_buffer.enabled and requested_track_policy == 'raw')
            track_path_position = 'before_mask_buffer' if track_before_mask_buffer else 'after_mask_buffer'
            track_raw_n = int(raw_n)
            mask_released_raw_n = 0
            mask_filtered_n = 0
            mask_release_consumed_for_viz_only = False
            event_viz_packet_pushed = False
            # 当前帧人体 mask 前置过滤：只使用同 sync_index 的当前 MVS 人体范围。
            # 但同一 MVS 帧约 40ms，会覆盖多个 4ms event slice，因此同 sync_index 内持久复用当前帧 mask。
            # 过滤后的 detection_evs 才进入 STC / spatter_tracker / line_filter / hit event。
            if event_human_mask_buffer.enabled:
                event_human_mask_buffer.submit(
                    raw_evs,
                    sync_index=current_sync_index_for_mask,
                    sync_event_ts_us=current_sync_event_ts_for_mask,
                    slice_ts_us=int(ts),
                    wall_time_us=int(capture_wall_time_us),
                    sync_diag=sync_stats_extra,
                )
                event_loop_profiler.mark('mask_buffer_submit')
                released = event_human_mask_buffer.pop_ready(
                    event_human_mask_filter,
                    now_wall_us=_now_wall_time_us(),
                )
                event_loop_profiler.mark('mask_buffer_pop_filter')
                if released is None:
                    try:
                        buffer_stats = event_human_mask_buffer.stats()
                    except Exception:
                        buffer_stats = {}
                    buffer_reason = str(buffer_stats.get('reason', 'human_mask_buffer_wait') or 'human_mask_buffer_wait')
                    if not buffer_reason.startswith('human_mask_buffer'):
                        buffer_reason = f'human_mask_buffer_{buffer_reason}'
                    if _push_event_viz_packet(
                        raw_evs, _empty_events_like(raw_evs), ts, capture_wall_time_us,
                        current_sync_index_for_mask, current_sync_event_ts_for_mask,
                        fastpath_reason=buffer_reason,
                        track_evs=None,
                        track_input_policy='human_mask_buffer_wait',
                    ):
                        event_viz_mainloop_stats['buffer_wait_heartbeat_pushed'] += 1
                    event_viz_packet_pushed = True
                    event_loop_profiler.mark('event_viz_buffer_wait_heartbeat')
                    if track_before_mask_buffer:
                        detection_evs = _empty_events_like(raw_evs)
                        detection_n = 0
                    else:
                        iter_idx += 1
                        _update_event_worker_status(
                            state='waiting',
                            active_reason='human_mask_buffer_wait',
                            ts_i=ts, raw_n_i=raw_n, detection_n_i=0,
                            raw_rate_mevs_i=0.0, fastpath=True, guard_state='BUFFER_WAIT',
                            track_n_i=0, track_input_policy_i='human_mask_buffer_wait',
                            track_path_position_i='after_mask_buffer',
                            track_raw_n_i=raw_n,
                            mask_buffer_blocks_tracking_i=True,
                            track_input_starved_reason_i=buffer_reason,
                            raw_evs_i=raw_evs,
                            track_evs_i=None,
                        )
                        event_loop_profiler.mark('status_buffer_wait')
                        if window is not None:
                            EventLoop.poll_and_dispatch()
                            if window.should_close():
                                break
                        event_loop_profiler.finish('buffer_wait', raw_n=raw_n, detection_n=0, fastpath=True)
                        continue
                if released is not None and track_before_mask_buffer:
                    mask_released_raw_n = int(released.get('raw_n') or 0)
                    mask_filtered_n = int(released.get('detection_n') or 0)
                    mask_release_consumed_for_viz_only = True
                    if _push_event_viz_packet(
                        released.get('raw_evs'), released.get('detection_evs'),
                        int(released.get('ts') or ts),
                        int(released.get('capture_wall_time_us') or capture_wall_time_us),
                        released.get('sync_index'), released.get('sync_event_ts_us'),
                        fastpath_reason=str(released.get('release_reason') or 'human_mask_buffer_release_viz_only'),
                        track_evs=raw_evs,
                        track_input_policy='raw_before_mask_buffer',
                    ):
                        event_viz_mainloop_stats['buffer_release_viz_only_pushed'] = (
                            int(event_viz_mainloop_stats.get('buffer_release_viz_only_pushed', 0)) + 1
                        )
                    event_viz_packet_pushed = True
                    detection_evs = released.get('detection_evs')
                    detection_n = int(mask_filtered_n)
                elif released is not None:
                    raw_evs = released.get('raw_evs')
                    detection_evs = released.get('detection_evs')
                    raw_n = int(released.get('raw_n') or 0)
                    detection_n = int(released.get('detection_n') or 0)
                    mask_released_raw_n = int(raw_n)
                    mask_filtered_n = int(detection_n)
                    ts = int(released.get('ts') or ts)
                    capture_wall_time_us = int(released.get('capture_wall_time_us') or capture_wall_time_us)
                    current_sync_index_for_mask = released.get('sync_index')
                    current_sync_event_ts_for_mask = released.get('sync_event_ts_us')
            else:
                detection_evs = event_human_mask_filter.filter_events(
                    raw_evs,
                    current_sync_index=current_sync_index_for_mask,
                    current_sync_event_ts_us=current_sync_event_ts_for_mask,
                    slice_ts_us=int(ts),
                    stats_extra=sync_stats_extra,
                )
                detection_n = int(len(detection_evs))
                mask_filtered_n = int(detection_n)
                event_loop_profiler.mark('mask_filter_direct')

            track_input_evs, track_n, track_input_policy, track_uses_raw = _select_track_input(raw_evs, detection_evs)
            if track_before_mask_buffer and track_uses_raw:
                track_input_policy = 'raw_before_mask_buffer'
            track_input_evs_for_viz = track_input_evs
            track_n_before_hot_pixel = int(track_n)
            hot_pixel_filter_stats = hot_pixel_filter.stats()
            if bool(getattr(args, 'event_hot_pixel_filter_enable', False)):
                track_input_evs, hot_pixel_filter_stats = hot_pixel_filter.filter(track_input_evs)
                try:
                    track_n = int(len(track_input_evs)) if track_input_evs is not None else 0
                except Exception:
                    track_n = 0
            event_loop_profiler.mark('track_hot_pixel_filter')
            _fastpath_reason = 'empty_slice' if raw_n <= 0 else ('track_input_empty' if track_n <= 0 else '')
            if (
                raw_n > 0 and track_n_before_hot_pixel > 0 and track_n <= 0
                and int(hot_pixel_filter_stats.get('removed_count', 0) or 0) > 0
            ):
                _fastpath_reason = 'hot_pixel_filter_removed_all'
            if raw_n > 0 and detection_n <= 0 and track_n > 0:
                _fastpath_reason = 'mask_filtered_empty_track_raw'
            if not event_viz_packet_pushed:
                _push_event_viz_packet(
                    raw_evs, detection_evs, ts, capture_wall_time_us,
                    current_sync_index_for_mask, current_sync_event_ts_for_mask,
                    fastpath_reason=_fastpath_reason,
                    track_evs=track_input_evs_for_viz,
                    track_input_policy=track_input_policy,
                )
            event_loop_profiler.mark('event_viz_push')

            if bool(getattr(args, 'event_empty_slice_fastpath_enable', True)) and (raw_n <= 0 or track_n <= 0):
                iter_idx += 1
                _update_event_worker_status(
                    state='idle',
                    active_reason='empty_slice' if raw_n <= 0 else 'track_input_empty',
                    ts_i=ts, raw_n_i=raw_n, detection_n_i=detection_n,
                    raw_rate_mevs_i=0.0, fastpath=True, guard_state='OK',
                    track_n_i=track_n, track_input_policy_i=track_input_policy,
                    track_path_position_i=track_path_position,
                    track_raw_n_i=track_raw_n,
                    mask_released_raw_n_i=mask_released_raw_n,
                    mask_filtered_n_i=mask_filtered_n,
                    mask_release_consumed_for_viz_only_i=mask_release_consumed_for_viz_only,
                    mask_buffer_blocks_tracking_i=bool(event_human_mask_buffer.enabled and not track_before_mask_buffer),
                    track_input_starved_reason_i=_fastpath_reason,
                    raw_evs_i=raw_evs,
                    track_evs_i=track_input_evs,
                    hot_pixel_filter_stats_i=hot_pixel_filter_stats,
                    track_n_before_hot_pixel_i=track_n_before_hot_pixel,
                )
                event_loop_profiler.mark('status_fastpath')
                if window is not None:
                    EventLoop.poll_and_dispatch()
                    if window.should_close():
                        break
                event_loop_profiler.finish(_fastpath_reason or 'fastpath', raw_n=raw_n, detection_n=detection_n, fastpath=True)
                continue

            if args.print_event_stats:
                if event_stats_last_ts is None:
                    event_stats_last_ts = int(ts)
                event_stats_acc_raw += raw_n
                dt_us_stats = int(ts) - int(event_stats_last_ts)
                if dt_us_stats >= event_stats_interval_us:
                    raw_mevs = float(event_stats_acc_raw) / float(max(1, dt_us_stats))
                    raw_itemsize = int(getattr(getattr(raw_evs, 'dtype', None), 'itemsize', 0) or 0)
                    raw_mib_s = (event_stats_acc_raw * raw_itemsize) / (dt_us_stats / 1_000_000.0) / (
                                1024.0 * 1024.0) if raw_itemsize > 0 else 0.0
                    now_wall_stats_us = _now_wall_time_us()
                    if event_stats_last_wall_us is None:
                        event_stats_last_wall_us = int(now_wall_stats_us)
                    wall_dt_us_stats = max(1, int(now_wall_stats_us) - int(event_stats_last_wall_us))
                    realtime_ratio = float(dt_us_stats) / float(wall_dt_us_stats)
                    lag_rate_ms = (float(wall_dt_us_stats) - float(dt_us_stats)) / 1000.0
                    print(
                        f"[EventPerf][{camera_log_tag}][SN={camera_sn_for_log}] "
                        f"raw={event_stats_acc_raw} "
                        f"event_dt_ms={dt_us_stats / 1000.0:.1f} "
                        f"wall_dt_ms={wall_dt_us_stats / 1000.0:.1f} "
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

            event_loop_profiler.mark('event_stats')

            dt_us_guard = int(args.accumulation_time_us) if last_loop_ts is None else max(1,
                                                                                          int(ts) - int(last_loop_ts))
            last_loop_ts = int(ts)
            shake_active, raw_rate_mevs, shake_entered, shake_exited = shake_guard.update(raw_n, dt_us_guard, ts)
            if shake_entered:
                print(
                    f"[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] ACTIVE raw_rate={raw_rate_mevs:.2f} MEv/s",
                    flush=True)
                # 进入保护时清空检测状态，避免全局抖动的垃圾轨迹污染后续结果
                _reset_realtime_detection_state('shake_guard_enter')
            elif shake_exited:
                print(
                    f"[ShakeGuard][{camera_log_tag}][SN={camera_sn_for_log}] RELEASE raw_rate={raw_rate_mevs:.2f} MEv/s",
                    flush=True)

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
                        release_by_raw = (pre_spatter_raw_thresh <= 0) or (
                                    raw_n <= int(pre_spatter_raw_thresh * pre_spatter_release_ratio))
                        release_by_rate = (pre_spatter_rate_thresh <= 0) or (
                                    raw_rate_mevs <= pre_spatter_rate_thresh * pre_spatter_release_ratio)
                        if release_by_time and release_by_raw and release_by_rate:
                            pre_spatter_guard_active = False
                            pre_spatter_over_count = 0
                            print(
                                f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] PRE_SPATTER_RELEASE "
                                f"raw_n={raw_n} raw_rate={raw_rate_mevs:.2f} MEv/s",
                                flush=True,
                            )

            raw_record_light_active = bool(
                getattr(args, 'raw_record_light_mode', False) and raw_recorder.is_recording())
            if raw_record_light_active and not raw_record_light_prev_active:
                print(f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] RAW_RECORD_LIGHT_ACTIVE", flush=True)
                _reset_realtime_detection_state('raw_record_light_enter')
            elif (not raw_record_light_active) and raw_record_light_prev_active:
                print(f"[RealtimeGuard][{camera_log_tag}][SN={camera_sn_for_log}] RAW_RECORD_LIGHT_RELEASE", flush=True)
            raw_record_light_prev_active = raw_record_light_active

            guard_active = bool(
                shake_active or cluster_guard_active or pre_spatter_guard_active or raw_record_light_active)
            event_loop_profiler.mark('guards')

            if guard_active:
                # 保护期间只保留最轻量的画面更新；重检测、子弹事件、TSF1 可按开关跳过
                if do_filtering:
                    stc_filter.process_events(track_input_evs, filtered_evs)
                buffer = filtered_evs if do_filtering else track_input_evs

                if tsf1_publisher is not None and (not args.shake_guard_skip_tsf1) and (
                not (pre_spatter_guard_active and args.pre_spatter_skip_tsf1)):
                    _maybe_publish_tsf1(buffer, ts, allow_guard=True)

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
                        event_overlay_pub_writer=_limited_overlay_writer(),
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
                    cv2.putText(output_img, guard_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    if window is not None:
                        window.show_async(output_img)
                    skip_obs_guard = bool((pre_spatter_guard_active and args.pre_spatter_skip_obs) or (
                                raw_record_light_active and args.raw_record_light_skip_obs))
                    if (not skip_obs_guard) and _limited_obs_submit(True):
                        _submit_obs_frame(
                            obs_streamer, output_img, line_filter, [],
                            obs_output_mode=args.obs_output_mode,
                            raw_event_img=raw_event_img,
                            obs_bullet_only=args.obs_bullet_only,
                            frame_ts_us=ts,
                        )
                if raw_record_light_active:
                    _guard_state_name = 'RAW_RECORD_LIGHT'
                elif cluster_guard_active and not shake_active:
                    _guard_state_name = 'CLUSTER_GUARD'
                elif pre_spatter_guard_active and not shake_active:
                    _guard_state_name = 'PRE_SPATTER'
                else:
                    _guard_state_name = 'SHAKE'
                _update_event_worker_status(
                    state='guard', active_reason=_guard_state_name,
                    ts_i=ts, raw_n_i=raw_n, detection_n_i=detection_n,
                    raw_rate_mevs_i=raw_rate_mevs, fastpath=False,
                    cluster_count_i=0, accepted_count_i=0, guard_state=_guard_state_name,
                    track_n_i=track_n, track_input_policy_i=track_input_policy,
                    track_path_position_i=track_path_position,
                    track_raw_n_i=track_raw_n,
                    mask_released_raw_n_i=mask_released_raw_n,
                    mask_filtered_n_i=mask_filtered_n,
                    mask_release_consumed_for_viz_only_i=mask_release_consumed_for_viz_only,
                    mask_buffer_blocks_tracking_i=False,
                    raw_evs_i=raw_evs,
                    track_evs_i=track_input_evs,
                    hot_pixel_filter_stats_i=hot_pixel_filter_stats,
                    track_n_before_hot_pixel_i=track_n_before_hot_pixel,
                )
                event_loop_profiler.mark('status_guard')
                if window is not None and window.should_close():
                    break
                event_loop_profiler.finish('guard', raw_n=raw_n, detection_n=detection_n, fastpath=False)
                continue

            if do_filtering:
                stc_filter.process_events(track_input_evs, filtered_evs)
            buffer = filtered_evs if do_filtering else track_input_evs
            event_loop_profiler.mark('stc_filter')

            if tsf1_publisher is not None:
                _maybe_publish_tsf1(buffer, ts, allow_guard=True)
            event_loop_profiler.mark('tsf1_pre_spatter')

            events_frame_gen_algo.process_events(buffer)
            event_loop_profiler.mark('frame_gen')

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
                        event_overlay_pub_writer=_limited_overlay_writer(),
                        event_overlay_pub_mode=args.event_overlay_pub_mode,
                    )
                    cv2.putText(output_img, "SPATTER_DISABLED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255),
                                2, cv2.LINE_AA)
                    if window is not None:
                        window.show_async(output_img)
                    if _limited_obs_submit(True):
                        _submit_obs_frame(
                            obs_streamer, output_img, line_filter, [],
                        obs_output_mode=args.obs_output_mode,
                        raw_event_img=raw_event_img,
                        obs_bullet_only=args.obs_bullet_only,
                        frame_ts_us=ts,
                    )
                _update_event_worker_status(
                    state='spatter_disabled', active_reason='disabled',
                    ts_i=ts, raw_n_i=raw_n, detection_n_i=detection_n,
                    raw_rate_mevs_i=raw_rate_mevs, fastpath=False,
                    cluster_count_i=0, accepted_count_i=0, guard_state='SPATTER_DISABLED',
                    track_n_i=track_n, track_input_policy_i=track_input_policy,
                    track_path_position_i=track_path_position,
                    track_raw_n_i=track_raw_n,
                    mask_released_raw_n_i=mask_released_raw_n,
                    mask_filtered_n_i=mask_filtered_n,
                    mask_release_consumed_for_viz_only_i=mask_release_consumed_for_viz_only,
                    mask_buffer_blocks_tracking_i=False,
                    raw_evs_i=raw_evs,
                    track_evs_i=track_input_evs,
                    hot_pixel_filter_stats_i=hot_pixel_filter_stats,
                    track_n_before_hot_pixel_i=track_n_before_hot_pixel,
                )
                event_loop_profiler.mark('status_spatter_disabled')
                if window is not None and window.should_close():
                    break
                event_loop_profiler.finish('spatter_disabled', raw_n=raw_n, detection_n=detection_n, fastpath=False)
                continue

            spatter_tracker.process_events(buffer, ts, clusters)
            event_loop_profiler.mark('spatter_tracker')

            clusters_np = clusters.numpy()
            cluster_count = int(len(clusters_np))
            clusters_np_for_line, cluster_sanitize_diag = _sanitize_clusters_for_line_filter(
                clusters_np, width, height,
                enabled=bool(getattr(args, 'event_cluster_sanitize_enable', True)),
            )
            track_polygon_gate_diag = {
                'mode': track_polygon_mode,
                'enabled': bool(track_polygon_gate is not None),
                'source': 'NONE',
                'reason': 'disabled' if track_polygon_mode == 'off' else 'camera_not_selected',
                'input_cluster_count': int(len(clusters_np_for_line)) if clusters_np_for_line is not None else 0,
                'gated_cluster_count': int(len(clusters_np_for_line)) if clusters_np_for_line is not None else 0,
                'filtered_cluster_count': 0,
            }
            polygon_shadow_clusters = clusters_np_for_line
            if track_polygon_gate is not None:
                clusters_np_for_line, polygon_shadow_clusters, track_polygon_gate_diag = track_polygon_gate.apply(
                    clusters_np_for_line,
                    ts=int(ts),
                    sync_index=current_sync_index_for_mask,
                    current_sync_event_ts_us=current_sync_event_ts_for_mask,
                )
            event_loop_profiler.mark('track_polygon_gate')
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
            event_loop_profiler.mark('clusters_numpy_count')

            if cluster_guard_enabled:
                if cluster_guard_count >= cluster_guard_thresh:
                    cluster_guard_over_count += 1
                else:
                    cluster_guard_over_count = 0

                if (not cluster_guard_active) and cluster_guard_over_count >= max(1,
                                                                                  int(args.shake_guard_trigger_slices)):
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
                            event_overlay_pub_writer=_limited_overlay_writer(),
                            event_overlay_pub_mode=args.event_overlay_pub_mode,
                        )
                        cv2.putText(output_img, "CLUSTER_GUARD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                                    2, cv2.LINE_AA)
                        if window is not None:
                            window.show_async(output_img)
                        if _limited_obs_submit(True):
                            _submit_obs_frame(
                                obs_streamer, output_img, line_filter, [],
                                obs_output_mode=args.obs_output_mode,
                                raw_event_img=raw_event_img,
                                obs_bullet_only=args.obs_bullet_only,
                                frame_ts_us=ts,
                            )
                    if window is not None and window.should_close():
                        break
                    event_loop_profiler.finish('cluster_guard_enter', raw_n=raw_n, detection_n=detection_n, fastpath=False)
                    continue

            accepted_clusters = line_filter.update(clusters_np_for_line, ts)
            event_loop_profiler.mark('line_filter')
            _debug_rows = line_filter.get_last_debug_rows()
            business_v1_assignments, business_id_v1_shadow_diag = business_id_v1_shadow.update(
                polygon_shadow_clusters,
                int(ts),
                track_polygon_gate_diag,
            )
            business_id_v3_shadow_diag = business_id_v3_shadow.update(
                business_v1_assignments,
                int(ts),
                track_polygon_gate_diag,
            )
            event_loop_profiler.mark('business_id_shadow')
            line_filter_shadow_diag = dict(shadow_line_filter_config) if isinstance(shadow_line_filter_config, dict) else {}
            line_filter_shadow_diag.update({
                'raw_cluster_count': int(cluster_count),
                'input_cluster_count': int(len(clusters_np_for_line)) if clusters_np_for_line is not None else 0,
                'sanitize_removed_count': int(cluster_sanitize_diag.get('removed_count', 0) or 0),
                'accepted_count': 0,
                'draw_paths_count': 0,
            })
            if shadow_line_filter is not None:
                try:
                    shadow_accepted = shadow_line_filter.update(clusters_np_for_line, ts)
                    shadow_debug_rows = (
                        shadow_line_filter.get_last_debug_rows()
                        if hasattr(shadow_line_filter, 'get_last_debug_rows') else []
                    )
                    shadow_draw_paths = (
                        shadow_line_filter.get_draw_paths(False)
                        if hasattr(shadow_line_filter, 'get_draw_paths') else []
                    )
                    line_filter_shadow_diag.update({
                        'accepted_count': int(len(shadow_accepted) if shadow_accepted is not None else 0),
                        'draw_paths_count': int(len(shadow_draw_paths) if shadow_draw_paths is not None else 0),
                        'line_filter_reject_diag': _line_filter_reject_diag(shadow_debug_rows),
                    })
                except Exception as exc:
                    line_filter_shadow_diag.update({
                        'error': f'{type(exc).__name__}: {exc}',
                    })
            _update_event_worker_status(
                state='active' if (raw_n > 0 or len(accepted_clusters) > 0) else 'idle',
                active_reason='accepted_cluster' if len(accepted_clusters) > 0 else 'events',
                ts_i=ts, raw_n_i=raw_n, detection_n_i=detection_n,
                raw_rate_mevs_i=raw_rate_mevs, fastpath=False,
                cluster_count_i=cluster_count, accepted_count_i=len(accepted_clusters),
                guard_state='OK',
                track_n_i=track_n, track_input_policy_i=track_input_policy,
                track_path_position_i=track_path_position,
                track_raw_n_i=track_raw_n,
                mask_released_raw_n_i=mask_released_raw_n,
                mask_filtered_n_i=mask_filtered_n,
                mask_release_consumed_for_viz_only_i=mask_release_consumed_for_viz_only,
                mask_buffer_blocks_tracking_i=False,
                raw_evs_i=raw_evs,
                track_evs_i=track_input_evs,
                clusters_np_i=clusters_np,
                accepted_clusters_i=accepted_clusters,
                debug_rows_i=_debug_rows,
                hot_pixel_filter_stats_i=hot_pixel_filter_stats,
                cluster_sanitize_diag_i=cluster_sanitize_diag,
                line_filter_shadow_diag_i=line_filter_shadow_diag,
                track_polygon_gate_diag_i=track_polygon_gate_diag,
                business_id_v1_shadow_diag_i=business_id_v1_shadow_diag,
                business_id_v3_shadow_diag_i=business_id_v3_shadow_diag,
                track_n_before_hot_pixel_i=track_n_before_hot_pixel,
            )
            event_loop_profiler.mark('status_active')

            for cluster in accepted_clusters:
                log.append([
                    ts,
                    int(cluster['bullet_id']), int(cluster['raw_id']),
                    int(cluster['x']), int(cluster['y']),
                    int(cluster['width']), int(cluster['height']),
                ])

            # 提取子弹事件并提交给发送器（主循环里只入队，不做网络 I/O）
            _write_spatter_raw_debug_rows(spatter_raw_debug_writer, ts, clusters_np, accepted_clusters, _debug_rows)
            spatter_summary_stats = _update_spatter_summary(
                spatter_summary_stats, ts, raw_n, cluster_count, len(accepted_clusters),
                int(getattr(args, 'spatter_summary_print_ms', 0) or 0),
                camera_log_tag, camera_sn_for_log,
            )
            try:
                bullet_extractor.set_event_context({
                    'event_human_mask_filter': event_human_mask_filter.get_last_stats(),
                    'event_human_mask_buffer': event_human_mask_buffer.stats(),
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
                    _refresh_bullet_display_diag(_debug_rows)
            except Exception as _e:
                try:
                    _refresh_bullet_display_diag(_debug_rows)
                except Exception:
                    pass
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
            event_loop_profiler.mark('bullet_extract_debug')

            iter_idx += 1
            need_render = (iter_idx % max(1, args.render_every_n) == 0)

            if need_render:
                _maybe_render(
                    ts, output_img, events_frame_gen_algo, line_filter,
                    accepted_clusters, input_nozone_rois, window, video_writer,
                    debug_draw_raw_clusters=args.debug_draw_raw_clusters,
                    raw_clusters_np=clusters_np,
                    raw_recording=raw_recorder.is_recording(),
                    obs_streamer=obs_streamer,
                    submit_obs=_limited_obs_submit(True),
                    obs_output_mode=args.obs_output_mode,
                    obs_bullet_only=args.obs_bullet_only,
                    out_video_mode=args.out_video_mode,
                    event_overlay_pub_writer=_limited_overlay_writer(),
                    event_overlay_pub_mode=args.event_overlay_pub_mode,
                )
            event_loop_profiler.mark('render')

            if window is not None and window.should_close():
                break
            event_loop_profiler.finish('active', raw_n=raw_n, detection_n=detection_n, fastpath=False)

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
        try:
            event_loop_profiler.flush(force=True)
        except Exception:
            pass
        try:
            _update_event_worker_status(state='stopping', active_reason='finally', force=True)
        except Exception:
            pass
        raw_record_control_service.close()
        raw_recorder.close()
        if video_writer is not None:
            video_writer.release()
            print('Video saved to', video_name)
        if event_viz_publisher is not None:
            try:
                event_viz_publisher.stop()
                event_viz_publisher.join(timeout=2.0)
            except Exception:
                pass
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
        try:
            event_sync_mapper.close()
        except Exception:
            pass
        if hasattr(mv_iterator, 'close'):
            try:
                mv_iterator.close()
            except Exception:
                pass
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
