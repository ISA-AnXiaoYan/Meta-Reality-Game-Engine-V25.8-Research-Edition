#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hik_mvs_obs_srt_streamer.py

海康机器人 / Hikrobot MVS SDK 工业相机直推 OBS 的 SRT 脚本。

链路：
  MVS SDK 直接取帧 -> OpenCV BGR 图 -> ffmpeg rawvideo 管道 -> H.264 / MPEG-TS / SRT -> OBS Media Source

典型运行：
  python hik_mvs_obs_srt_streamer.py \
    --serial DA7653808 \
    --srt-port 6201 \
    --out-width 1280 --out-height 720 \
    --fps 25 \
    --bitrate-kbps 3500 \
    --show-local

OBS 媒体源输入：
  srt://采集机IP:6201?mode=caller&transtype=live&latency=80000&tlpktdrop=1
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import statistics
import threading
import queue
from ctypes import *
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_MVIMPORT = "/opt/MVS/Samples/64/Python/MvImport"


def _add_mvs_import_path(mvimport_path: str) -> None:
    p = Path(mvimport_path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 MVS Python MvImport 路径: {mvimport_path}\n"
            "请用下面命令确认路径：\n"
            "  find /opt/MVS -name MvCameraControl_class.py\n"
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class FfmpegSrtWriter:
    """把 BGR24 原始帧通过 stdin 喂给 ffmpeg，再由 ffmpeg 推 SRT；支持断开后自动重连。"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        port: int,
        bitrate_kbps: int = 3500,
        latency_ms: int = 80,
        bind_ip: str = "0.0.0.0",
        loglevel: str = "warning",
        reconnect_delay_s: float = 2.0,
        tlpktdrop: bool = False,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.port = int(port)
        self.bitrate_kbps = int(bitrate_kbps)
        self.latency_ms = int(latency_ms)
        self.bind_ip = str(bind_ip)
        self.loglevel = str(loglevel)
        self.reconnect_delay_s = max(0.1, float(reconnect_delay_s))
        self.proc: Optional[subprocess.Popen] = None
        self.frames_written = 0
        self._next_restart_wall = 0.0
        self._last_exit_report_wall = 0.0
        # OBS 同步优先：默认不丢迟到包。低延迟优先时可显式开启。
        self.tlpktdrop = bool(tlpktdrop)

    def _build_url(self) -> str:
        latency_us = max(1, int(self.latency_ms) * 1000)
        return (
            f"srt://{self.bind_ip}:{self.port}"
            f"?mode=listener"
            f"&transtype=live"
            f"&latency={latency_us}"
            f"&tlpktdrop={1 if self.tlpktdrop else 0}"
            f"&pkt_size=1316"
        )

    def _cleanup_proc(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
        except Exception:
            pass

    def _schedule_restart(self) -> None:
        self._cleanup_proc()
        self._next_restart_wall = time.time() + self.reconnect_delay_s

    def start(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        if self.proc is not None and self.proc.poll() is not None:
            rc = self.proc.returncode
            now = time.time()
            if now - self._last_exit_report_wall > 0.5:
                print(f"[HIK-SRT] ffmpeg exited with code={rc}; will reconnect in {self.reconnect_delay_s:.1f}s", flush=True)
                self._last_exit_report_wall = now
            self._schedule_restart()
            return False
        now = time.time()
        if now < self._next_restart_wall:
            return False
        url = self._build_url()
        gop = max(1, int(round(self.fps)))
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", self.loglevel,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-b:v", f"{self.bitrate_kbps}k",
            "-maxrate", f"{self.bitrate_kbps}k", "-bufsize", f"{max(500, self.bitrate_kbps // 2)}k",
            "-g", str(gop), "-muxdelay", "0", "-muxpreload", "0",
            "-f", "mpegts", url,
        ]
        print("[HIK-SRT] start ffmpeg:")
        print(" ".join(cmd), flush=True)
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            return True
        except Exception as exc:
            print(f"[HIK-SRT] start ffmpeg failed: {exc}; will retry in {self.reconnect_delay_s:.1f}s", flush=True)
            self._next_restart_wall = time.time() + self.reconnect_delay_s
            return False

    def write(self, frame_bgr: np.ndarray) -> bool:
        if not self.start():
            return False
        if self.proc is None or self.proc.stdin is None:
            return False
        if self.proc.poll() is not None:
            self.start()
            return False
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame size mismatch: got {frame_bgr.shape}, expected {(self.height, self.width, 3)}")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected BGR frame HxWx3, got {frame_bgr.shape}")
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
            self.frames_written += 1
            return True
        except BrokenPipeError:
            print(f"[HIK-SRT] ffmpeg stdin broken pipe; will reconnect in {self.reconnect_delay_s:.1f}s", flush=True)
            self._schedule_restart()
            return False
        except OSError as exc:
            print(f"[HIK-SRT] ffmpeg write error: {exc}; will reconnect in {self.reconnect_delay_s:.1f}s", flush=True)
            self._schedule_restart()
            return False

    def close(self) -> None:
        self._cleanup_proc()
        print(f"[HIK-SRT] stopped, frames_written={self.frames_written}", flush=True)


class AsyncFfmpegSrtWriter:
    """异步写 ffmpeg/SRT：采集主循环只 submit，不直接阻塞在 stdin.write。"""

    def __init__(self, inner: FfmpegSrtWriter, queue_size: int = 4) -> None:
        self.inner = inner
        self.queue_size = max(1, int(queue_size))
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.submitted = 0
        self.dropped_old = 0
        self.write_failed = 0
        self._last_report_wall = 0.0

    @property
    def frames_written(self) -> int:
        return int(getattr(self.inner, "frames_written", 0))

    def start(self) -> bool:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="MVS-Async-SRT-Writer", daemon=True)
            self._thread.start()
            print(f"[HIK-SRT][ASYNC] enabled: queue_size={self.queue_size}", flush=True)
        return True

    def submit(self, frame_bgr: np.ndarray) -> bool:
        self.start()
        if frame_bgr is None:
            return False
        # 必须 copy：采集循环会释放/复用底层 buffer，异步线程不能引用临时数组。
        frame = frame_bgr.copy()
        self.submitted += 1
        try:
            self.q.put_nowait(frame)
            return True
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.dropped_old += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(frame)
                return True
            except queue.Full:
                self.dropped_old += 1
                return False

    def write(self, frame_bgr: np.ndarray) -> bool:
        return self.submit(frame_bgr)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            ok = self.inner.write(frame)
            if not ok:
                self.write_failed += 1
                time.sleep(0.02)
            now = time.time()
            if now - self._last_report_wall >= 5.0:
                if self.dropped_old > 0 or self.write_failed > 0:
                    print(
                        f"[HIK-SRT][ASYNC] submitted={self.submitted} written={self.frames_written} "
                        f"queue={self.q.qsize()}/{self.queue_size} dropped_old={self.dropped_old} "
                        f"write_failed={self.write_failed}",
                        flush=True,
                    )
                self._last_report_wall = now

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.inner.close()
        print(
            f"[HIK-SRT][ASYNC] stopped, submitted={self.submitted}, "
            f"written={self.frames_written}, dropped_old={self.dropped_old}, write_failed={self.write_failed}",
            flush=True,
        )


class TimestampAlignedFrameSender:
    """v3：按 event_ts_us 对齐输出的帧发送器。

    上游 submit(ts_us, frame) 只把帧放入 timestamp buffer；后台线程按固定 fps 输出
    target_ts = latest_ts_us - delay_us 附近的历史帧。这样 MVS 路和事件相机路可以用
    同一个 event_ts_us 时间轴进入 OBS，而不是各自按 wall-clock 立即推流。
    """

    def __init__(self, inner, fps: float, delay_ms: int = 1000, buffer_frames: int = 250) -> None:
        self.inner = inner
        self.fps = max(0.1, float(fps))
        self.delay_us = max(0, int(delay_ms) * 1000)
        self.buffer_frames = max(10, int(buffer_frames))
        self._cond = threading.Condition()
        self._buf: List[Tuple[int, np.ndarray]] = []
        self._latest_ts_us: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.submitted = 0
        self.output = 0
        self.dropped_old = 0
        self.missing_ts = 0
        self._last_report_wall = 0.0

    @property
    def frames_written(self) -> int:
        return int(getattr(self.inner, "frames_written", 0))

    def start(self) -> bool:
        try:
            if hasattr(self.inner, "start"):
                self.inner.start()
        except Exception:
            pass
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="MVS-TS-Aligned-SRT", daemon=True)
            self._thread.start()
            print(
                f"[MVS][OBS-TS-ALIGN] enabled: delay_ms={self.delay_us/1000.0:.1f}, "
                f"fps={self.fps:.3f}, buffer_frames={self.buffer_frames}",
                flush=True,
            )
        return True

    def submit(self, ts_us: int, frame_bgr: np.ndarray) -> bool:
        self.start()
        if frame_bgr is None:
            return False
        try:
            ts_i = int(ts_us)
        except Exception:
            self.missing_ts += 1
            return False
        if ts_i < 0:
            self.missing_ts += 1
            return False
        frame = frame_bgr.copy()
        with self._cond:
            if not self._buf or ts_i >= self._buf[-1][0]:
                self._buf.append((ts_i, frame))
            else:
                inserted = False
                for i, (old_ts, _old_frame) in enumerate(self._buf):
                    if ts_i < old_ts:
                        self._buf.insert(i, (ts_i, frame))
                        inserted = True
                        break
                if not inserted:
                    self._buf.append((ts_i, frame))
            self._latest_ts_us = ts_i if self._latest_ts_us is None else max(self._latest_ts_us, ts_i)
            while len(self._buf) > self.buffer_frames:
                self._buf.pop(0)
                self.dropped_old += 1
            self.submitted += 1
            self._cond.notify()
        return True

    def write(self, frame_bgr: np.ndarray) -> bool:
        # 兼容旧调用：没有 timestamp 时不输出，避免破坏对齐。
        self.missing_ts += 1
        return False

    def _pop_locked(self) -> Optional[np.ndarray]:
        if not self._buf or self._latest_ts_us is None:
            return None
        target_ts = int(self._latest_ts_us) - int(self.delay_us)
        if self._buf[0][0] > target_ts:
            return None
        idx = None
        for i, (ts_i, _frame) in enumerate(self._buf):
            if ts_i <= target_ts:
                idx = i
            else:
                break
        if idx is None:
            return None
        _ts_i, frame = self._buf[idx]
        del self._buf[:idx + 1]
        return frame

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        next_tick = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_tick:
                with self._cond:
                    self._cond.wait(timeout=min(next_tick - now, 0.2))
                continue
            with self._cond:
                frame = self._pop_locked()
            if frame is None:
                with self._cond:
                    self._cond.wait(timeout=0.02)
                next_tick = max(next_tick, time.time())
                continue
            ok = self.inner.write(frame)
            if ok:
                self.output += 1
            next_tick += interval
            if next_tick < time.time() - interval:
                next_tick = time.time() + interval
            now2 = time.time()
            if now2 - self._last_report_wall >= 5.0:
                if self.dropped_old > 0 or self.missing_ts > 0:
                    print(
                        f"[MVS][OBS-TS-ALIGN] submitted={self.submitted} output={self.output} "
                        f"written={self.frames_written} buffer={len(self._buf)}/{self.buffer_frames} "
                        f"dropped_old={self.dropped_old} missing_ts={self.missing_ts}",
                        flush=True,
                    )
                self._last_report_wall = now2

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.inner.close()
        except Exception:
            pass
        print(
            f"[MVS][OBS-TS-ALIGN] stopped, submitted={self.submitted}, output={self.output}, "
            f"written={self.frames_written}, dropped_old={self.dropped_old}, missing_ts={self.missing_ts}",
            flush=True,
        )


def resize_fit(frame: np.ndarray, out_w: int, out_h: int, mode: str = "contain") -> np.ndarray:
    """把相机帧变成指定输出尺寸。contain 保持比例加黑边，crop 保持比例裁切，stretch 直接拉伸。"""
    out_w = int(out_w)
    out_h = int(out_h)
    h, w = frame.shape[:2]

    if mode == "stretch":
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    if mode == "crop":
        scale = max(out_w / max(1, w), out_h / max(1, h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = max(0, (nw - out_w) // 2)
        y0 = max(0, (nh - out_h) // 2)
        return resized[y0:y0 + out_h, x0:x0 + out_w].copy()

    scale = min(out_w / max(1, w), out_h / max(1, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _bytes_from_c_array(arr) -> bytes:
    return bytes(bytearray(arr)).split(b"\x00", 1)[0]


def parse_args():
    ap = argparse.ArgumentParser(description="Hikrobot MVS SDK camera to OBS SRT streamer")
    ap.add_argument("--mvimport-path", default=DEFAULT_MVIMPORT, help="MvCameraControl_class.py 所在目录")
    ap.add_argument("--serial", default="", help="相机序列号，例如 DA7653808；不填则打开第一个")
    ap.add_argument("--index", type=int, default=0, help="未指定 serial 时打开第几个设备")
    ap.add_argument("--list-devices", action="store_true", help="只列出设备后退出")

    ap.add_argument("--srt-port", type=int, default=6201, help="SRT listener 端口")
    ap.add_argument("--bind-ip", default="0.0.0.0", help="SRT 绑定地址")
    ap.add_argument("--latency", type=int, default=80, help="SRT latency，单位毫秒；脚本内部会转换成 ffmpeg/libsrt 需要的微秒")
    ap.add_argument("--reconnect-delay", type=float, default=2.0, help="ffmpeg/SRT 断开后等待多少秒自动重连")
    ap.add_argument("--tlpktdrop", dest="tlpktdrop", action="store_true", default=False,
                    help="SRT 开启丢弃迟到包，低延迟优先但可能破坏两路画面完整性/同步")
    ap.add_argument("--no-tlpktdrop", dest="tlpktdrop", action="store_false",
                    help="SRT 不丢弃迟到包，同步/完整性优先；推荐同步调试时使用")
    ap.add_argument("--async-srt", dest="async_srt", action="store_true", default=True,
                    help="异步写 ffmpeg/SRT，避免 writer.write 阻塞 MVS 采集主循环；默认开启")
    ap.add_argument("--no-async-srt", dest="async_srt", action="store_false",
                    help="关闭异步 SRT，退回采集循环内同步写 ffmpeg")
    ap.add_argument("--async-queue-size", type=int, default=4,
                    help="异步 SRT 队列大小；满时丢旧帧保最新，避免采集阻塞")
    ap.add_argument("--obs-align-by-ts", dest="obs_align_by_ts", action="store_true", default=False,
                    help="v3：MVS OBS 输出按 event_ts_us 缓冲对齐；需要 event_trigger_log 和 sync_offset")
    ap.add_argument("--obs-no-align-by-ts", dest="obs_align_by_ts", action="store_false",
                    help="关闭 MVS OBS timestamp 对齐，恢复实时最新帧输出")
    ap.add_argument("--obs-align-delay-ms", type=int, default=1000,
                    help="v3：OBS timestamp 对齐延迟，单位 ms。事件相机和 MVS 应使用相同值")
    ap.add_argument("--obs-align-buffer-frames", type=int, default=250,
                    help="v3：MVS OBS timestamp 对齐缓冲帧数上限")
    ap.add_argument("--obs-align-event-reload-s", type=float, default=0.10,
                    help="v3：MVS 对齐模式下读取 event_trigger CSV 的最小间隔，秒")
    ap.add_argument("--fps", type=float, default=25.0, help="推流 FPS")
    ap.add_argument("--bitrate-kbps", type=int, default=3500, help="视频码率 kbps")
    ap.add_argument("--out-width", type=int, default=1280, help="OBS 输出宽")
    ap.add_argument("--out-height", type=int, default=720, help="OBS 输出高")
    ap.add_argument("--fit", choices=["contain", "crop", "stretch"], default="contain", help="缩放方式")

    ap.add_argument("--exposure-us", type=float, default=None, help="曝光时间 us；不填不改")
    ap.add_argument("--gain", type=float, default=None, help="增益；不填不改")
    ap.add_argument("--camera-fps", type=float, default=None, help="相机 AcquisitionFrameRate；不填不改")

    ap.add_argument("--enable-sync-output", action="store_true", help="启用 MVS Line 输出同步脉冲")
    ap.add_argument("--sync-line-selector", default="Line1", help="同步输出线路，例如 Line1")
    ap.add_argument("--sync-line-mode", default="Strobe", help="线路模式，例如 Strobe")
    ap.add_argument("--sync-line-source", default="ExposureStartActive", help="线路源，例如 ExposureStartActive")
    ap.add_argument("--sync-line-inverter", action="store_true", help="是否翻转输出线路")
    ap.add_argument("--sync-output-duration-us", type=float, default=1000.0, help="输出脉冲持续时间 us")
    ap.add_argument("--sync-output-delay-us", type=float, default=0.0, help="输出延迟 us")
    ap.add_argument("--sync-output-pre-delay-us", type=float, default=0.0, help="输出预延迟 us")
    ap.add_argument("--mvs-frame-sync-log", default="", help="MVS 每个采集帧的同步日志 CSV 路径")
    ap.add_argument("--wait-ready-file", default="", help="等待事件相机 ready 标志文件出现后再 StartGrabbing")
    ap.add_argument("--wait-ready-timeout-s", type=float, default=20.0, help="等待 ready 文件超时时间，秒；<=0 表示一直等待")
    ap.add_argument("--wait-ready-extra-delay-s", type=float, default=0.2, help="ready 文件出现后额外等待时间，秒")

    # MVS 启动预热/稳定后正式同步记录：
    # 启动初期可能因为 ffmpeg/显示/SDK 缓冲导致 mvs_frame_num 跳号。
    # 这里允许先只快速取帧并丢弃，不推流、不显示；达到 warmup + 连续稳定帧数后，
    # 再开始正式推流和正式同步日志。
    ap.add_argument("--sync-warmup-seconds", type=float, default=0.0,
                    help="MVS StartGrabbing 后预热多少秒；预热期只取帧释放，不推流/显示。0=关闭时间预热")
    ap.add_argument("--sync-stable-min-frames", type=int, default=1,
                    help="正式同步前要求最近连续多少个 MVS 帧号无异常跳号。1=不等待连续稳定帧")
    ap.add_argument("--sync-stable-gap-tolerance", type=int, default=0,
                    help="判断稳定时允许的 mvs_frame_num 缺帧数；0 表示必须严格连续")
    ap.add_argument("--sync-official-log", default="",
                    help="稳定后正式 MVS 同步日志 CSV；只记录可用于后续对齐/检测的 MVS 帧")

    # v5：自动估计 MVS frame_num 与事件相机 sync_index 的 offset，并输出带 event_ts_us 的正式配对表。
    # 关系定义：event_sync_index = mvs_frame_num + sync_offset
    ap.add_argument("--event-trigger-log", default="",
                    help="事件相机 External Trigger CSV 路径，用于读取 sync_index/event_ts_us 并做自动配对")
    ap.add_argument("--sync-paired-official-log", default="",
                    help="稳定后正式配对日志 CSV；自动写入 mvs_frame_num 对应的 event_sync_index/event_ts_us")
    ap.add_argument("--sync-fixed-offset", type=int, default=None,
                    help="手动指定 event_sync_index = mvs_frame_num + offset；不填则自动估计")
    ap.add_argument("--sync-offset-search", type=int, default=5,
                    help="自动估计 offset 时搜索 [-N, +N]，关系为 event_sync_index=mvs_frame_num+offset")
    ap.add_argument("--sync-offset-min-pairs", type=int, default=20,
                    help="自动锁定 offset 至少需要多少个可配对样本")
    ap.add_argument("--sync-offset-refresh-s", type=float, default=0.5,
                    help="读取事件 trigger CSV 并尝试估计 offset 的间隔，秒")
    ap.add_argument("--sync-offset-max-wall-diff-ms", type=float, default=20.0,
                    help="自动 offset 接受阈值：MVS wall_time 与 event wall_time 的中位绝对差需小于该值，毫秒")

    ap.add_argument("--show-local", action="store_true", help="本地 OpenCV 预览")
    ap.add_argument("--print-every", type=int, default=100, help="每 N 帧打印一次状态")
    return ap.parse_args()


def main():
    args = parse_args()
    _add_mvs_import_path(args.mvimport_path)

    # 这里放在 main 中 import，方便先改 mvimport-path
    from MvCameraControl_class import (  # noqa
        MvCamera,
        MV_CC_DEVICE_INFO_LIST,
        MV_CC_DEVICE_INFO,
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
        MV_ACCESS_Exclusive,
        MV_TRIGGER_MODE_OFF,
        MV_FRAME_OUT,
        MV_CC_PIXEL_CONVERT_PARAM_EX,
        PixelType_Gvsp_BGR8_Packed,
    )

    def device_serial(dev_info) -> str:
        try:
            if dev_info.nTLayerType == MV_USB_DEVICE:
                return _bytes_from_c_array(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode(errors="ignore")
            if dev_info.nTLayerType == MV_GIGE_DEVICE:
                return _bytes_from_c_array(dev_info.SpecialInfo.stGigEInfo.chSerialNumber).decode(errors="ignore")
        except Exception:
            pass
        return ""

    def device_model(dev_info) -> str:
        try:
            if dev_info.nTLayerType == MV_USB_DEVICE:
                return _bytes_from_c_array(dev_info.SpecialInfo.stUsb3VInfo.chModelName).decode(errors="ignore")
            if dev_info.nTLayerType == MV_GIGE_DEVICE:
                return _bytes_from_c_array(dev_info.SpecialInfo.stGigEInfo.chModelName).decode(errors="ignore")
        except Exception:
            pass
        return ""

    def enum_devices():
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed, ret=0x{ret:x}")
        devices = []
        for i in range(int(device_list.nDeviceNum)):
            dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            devices.append((i, dev_info, device_serial(dev_info), device_model(dev_info)))
        return device_list, devices

    def select_device(device_list, serial: str = "", index: Optional[int] = None):
        _, devices = enum_devices()
        if not devices:
            raise RuntimeError("没有发现海康 MVS 相机。请确认 MVS 软件能看到设备，并且没有独占占用。")
        print("[MVS] detected devices:")
        for i, _dev, sn, model in devices:
            print(f"  [{i}] serial={sn} model={model}")
        if serial:
            serial = str(serial).strip()
            for i, dev_info, sn, _model in devices:
                if sn == serial:
                    return cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            raise RuntimeError(f"未找到 serial={serial} 的相机")
        if index is None:
            index = 0
        if index < 0 or index >= len(devices):
            raise RuntimeError(f"index={index} 越界，当前设备数={len(devices)}")
        return cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

    def set_if_possible(cam, kind: str, name: str, value):
        try:
            if value is None:
                return
            if kind == "float":
                ret = cam.MV_CC_SetFloatValue(name, float(value))
            elif kind == "int":
                ret = cam.MV_CC_SetIntValue(name, int(value))
            elif kind == "enum":
                ret = cam.MV_CC_SetEnumValue(name, int(value))
            elif kind == "bool":
                ret = cam.MV_CC_SetBoolValue(name, bool(value))
            else:
                return
            if ret != 0:
                print(f"[MVS][WARN] set {name}={value} failed ret=0x{ret:x}")
            else:
                print(f"[MVS] set {name}={value}")
        except Exception as exc:
            print(f"[MVS][WARN] set {name}={value} exception: {exc}")

    def set_enum_string_if_possible(cam, name: str, value: str):
        try:
            if value is None:
                return False
            if not hasattr(cam, "MV_CC_SetEnumValueByString"):
                print(f"[MVS][WARN] SDK has no MV_CC_SetEnumValueByString; cannot set {name}={value}", flush=True)
                return False
            ret = cam.MV_CC_SetEnumValueByString(name, str(value))
            if ret != 0:
                print(f"[MVS][WARN] set {name}={value} failed ret=0x{ret:x}", flush=True)
                return False
            print(f"[MVS] set {name}={value}", flush=True)
            return True
        except Exception as exc:
            print(f"[MVS][WARN] set {name}={value} exception: {exc}", flush=True)
            return False

    def configure_mvs_sync_output(cam, args):
        if not getattr(args, "enable_sync_output", False):
            print("[MVS][SYNC] sync output disabled", flush=True)
            return
        print("[MVS][SYNC] configuring Line output for frame sync...", flush=True)
        set_enum_string_if_possible(cam, "LineSelector", args.sync_line_selector)
        set_enum_string_if_possible(cam, "LineMode", args.sync_line_mode)
        set_enum_string_if_possible(cam, "LineSource", args.sync_line_source)
        set_if_possible(cam, "bool", "LineInverter", bool(args.sync_line_inverter))
        # GUI 里的“输出使能”通常对应 StrobeEnable。
        set_if_possible(cam, "bool", "StrobeEnable", True)
        set_if_possible(cam, "float", "StrobeLineDuration", float(args.sync_output_duration_us))
        set_if_possible(cam, "float", "StrobeLineDelay", float(args.sync_output_delay_us))
        set_if_possible(cam, "float", "StrobeLinePreDelay", float(args.sync_output_pre_delay_us))
        print("[MVS][SYNC] sync output config done", flush=True)

    def wait_ready_file_if_needed(args):
        ready_file = str(getattr(args, "wait_ready_file", "") or "").strip()
        if not ready_file:
            return
        deadline = None
        timeout = float(getattr(args, "wait_ready_timeout_s", 20.0) or 0.0)
        if timeout > 0:
            deadline = time.time() + timeout
        print(f"[MVS][SYNC] waiting event ready file before StartGrabbing: {ready_file}", flush=True)
        while not os.path.exists(ready_file):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"wait ready file timeout: {ready_file}")
            time.sleep(0.05)
        extra = max(0.0, float(getattr(args, "wait_ready_extra_delay_s", 0.2) or 0.0))
        if extra > 0:
            time.sleep(extra)
        print(f"[MVS][SYNC] event ready file detected; StartGrabbing now: {ready_file}", flush=True)

    def convert_frame_to_bgr(cam, frame_out) -> np.ndarray:
        info = frame_out.stFrameInfo
        width = int(info.nWidth)
        height = int(info.nHeight)
        src_len = int(info.nFrameLen)

        dst_size = width * height * 3
        dst_buf = (c_ubyte * dst_size)()

        conv = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(conv), 0, sizeof(conv))

        conv.nWidth = width
        conv.nHeight = height
        conv.pSrcData = cast(frame_out.pBufAddr, POINTER(c_ubyte))
        conv.nSrcDataLen = src_len
        conv.enSrcPixelType = info.enPixelType
        conv.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        conv.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        conv.nDstBufferSize = dst_size

        ret = cam.MV_CC_ConvertPixelTypeEx(conv)
        if ret != 0:
            raise RuntimeError(f"MV_CC_ConvertPixelTypeEx failed, ret=0x{ret:x}, src_pixel_type={info.enPixelType}")

        arr = np.frombuffer(dst_buf, dtype=np.uint8, count=dst_size)
        return arr.reshape((height, width, 3)).copy()

    def _safe_int(value, default: int = -1) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def load_event_trigger_rows(path: str) -> Dict[int, Dict[str, int]]:
        """读取事件相机 trigger CSV。

        注意：事件相机进程还在写文件时，最后一行可能是不完整的，所以这里全部做容错。
        返回 dict: sync_index -> row。只保留 sync_index >= 0 的选中 polarity trigger。
        """
        event_by_sync: Dict[int, Dict[str, int]] = {}
        if not path:
            return event_by_sync
        p = Path(path)
        if not p.exists():
            return event_by_sync
        try:
            with p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    si = _safe_int(r.get("sync_index", -1), -1)
                    if si < 0:
                        continue
                    event_by_sync[si] = {
                        "all_index": _safe_int(r.get("all_index", -1), -1),
                        "sync_index": si,
                        "event_ts_us": _safe_int(r.get("event_ts_us", -1), -1),
                        "polarity": _safe_int(r.get("polarity", -1), -1),
                        "trigger_id": _safe_int(r.get("trigger_id", -1), -1),
                        "slice_ts_us": _safe_int(r.get("slice_ts_us", -1), -1),
                        "wall_time_us": _safe_int(r.get("wall_time_us", -1), -1),
                    }
        except Exception as exc:
            print(f"[MVS][SYNC][WARN] read event trigger log failed: {path}: {exc}", flush=True)
        return event_by_sync

    def estimate_sync_offset(
        recent_mvs_rows: List[Dict[str, int]],
        event_by_sync: Dict[int, Dict[str, int]],
        search_n: int,
        min_pairs: int,
        max_wall_diff_us: int,
    ) -> Optional[Tuple[int, int, float, float]]:
        """自动估计 offset。关系：event_sync_index = mvs_frame_num + offset。

        返回 (offset, pairs, median_abs_wall_diff_us, median_signed_wall_diff_us)。
        """
        if not recent_mvs_rows or not event_by_sync:
            return None
        search_n = max(0, int(search_n))
        min_pairs = max(1, int(min_pairs))
        candidates = []
        for offset in range(-search_n, search_n + 1):
            diffs = []
            for m in recent_mvs_rows[-300:]:
                mvs_frame_num = int(m.get("mvs_frame_num", -1))
                mvs_wall = int(m.get("mvs_wall_time_us", -1))
                if mvs_frame_num < 0 or mvs_wall <= 0:
                    continue
                e = event_by_sync.get(mvs_frame_num + offset)
                if not e:
                    continue
                event_wall = int(e.get("wall_time_us", -1))
                if event_wall <= 0:
                    continue
                diffs.append(mvs_wall - event_wall)
            if len(diffs) < min_pairs:
                continue
            abs_diffs = [abs(x) for x in diffs]
            med_abs = float(statistics.median(abs_diffs))
            med_signed = float(statistics.median(diffs))
            candidates.append((offset, len(diffs), med_abs, med_signed))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[2], -x[1], abs(x[0])))
        best = candidates[0]
        if max_wall_diff_us > 0 and best[2] > max_wall_diff_us:
            return None
        return best

    def estimate_event_interval_us(event_by_sync: Dict[int, Dict[str, int]], fallback_us: int) -> int:
        """从最近的 trigger 行估计事件时间轴帧间隔。"""
        try:
            keys = sorted(event_by_sync.keys())[-200:]
            diffs = []
            prev_k = None
            prev_ts = None
            for k in keys:
                ts_i = int(event_by_sync[k].get("event_ts_us", -1))
                if prev_k is not None and prev_ts is not None and k == prev_k + 1 and ts_i > 0 and prev_ts > 0:
                    d = ts_i - prev_ts
                    if 1_000 <= d <= 200_000:
                        diffs.append(d)
                prev_k = k
                prev_ts = ts_i
            if diffs:
                return int(statistics.median(diffs))
        except Exception:
            pass
        return int(fallback_us)

    def get_event_ts_for_sync_index(
        event_by_sync: Dict[int, Dict[str, int]],
        sync_index: int,
        interval_us: int,
    ) -> Tuple[int, str]:
        """返回 sync_index 对应的 event_ts_us。

        优先使用 CSV 中的精确 trigger timestamp；如果当前行还没 flush 到 CSV，
        就用最近一个已知 trigger 做线性估计，避免 MVS OBS 对齐输出等待 CSV 而卡顿。
        """
        try:
            si = int(sync_index)
        except Exception:
            return -1, "bad_index"
        e = event_by_sync.get(si)
        if e is not None:
            ts_i = int(e.get("event_ts_us", -1))
            if ts_i >= 0:
                return ts_i, "exact"
        if not event_by_sync:
            return -1, "no_event_rows"
        try:
            # 选离目标 sync_index 最近的 anchor，比固定用最新 anchor 更稳。
            anchor_idx = min(event_by_sync.keys(), key=lambda k: abs(int(k) - si))
            anchor_ts = int(event_by_sync[anchor_idx].get("event_ts_us", -1))
            if anchor_ts < 0:
                return -1, "bad_anchor"
            return int(anchor_ts + (si - int(anchor_idx)) * int(interval_us)), "estimated"
        except Exception:
            return -1, "estimate_failed"

    def try_write_paired_rows(
        pending_rows: List[Dict[str, int]],
        event_by_sync: Dict[int, Dict[str, int]],
        paired_writer,
        paired_file,
        sync_offset: Optional[int],
        paired_counter: int,
    ) -> int:
        """把已经进入 official 阶段的 MVS 帧补写成 MVS-event 配对日志。

        如果事件 CSV 因 flush 滞后暂时没有对应 sync_index，就先保留在 pending_rows，
        下一轮重新读取 event_trigger_log 后再补写。
        """
        if paired_writer is None or sync_offset is None or not pending_rows:
            return paired_counter

        remain: List[Dict[str, int]] = []
        wrote = 0
        for m in pending_rows:
            mvs_frame_num = int(m.get("mvs_frame_num", -1))
            event_sync_index = mvs_frame_num + int(sync_offset)
            e = event_by_sync.get(event_sync_index)
            if e is None:
                remain.append(m)
                continue
            mvs_wall = int(m.get("mvs_wall_time_us", -1))
            event_wall = int(e.get("wall_time_us", -1))
            paired_writer.writerow({
                "paired_index": int(paired_counter),
                "is_official": 1,
                "sync_offset": int(sync_offset),
                "mvs_frame_index": int(m.get("mvs_frame_index", -1)),
                "mvs_frame_num": int(mvs_frame_num),
                "event_sync_index": int(event_sync_index),
                "event_ts_us": int(e.get("event_ts_us", -1)),
                "event_all_index": int(e.get("all_index", -1)),
                "event_polarity": int(e.get("polarity", -1)),
                "event_trigger_id": int(e.get("trigger_id", -1)),
                "event_slice_ts_us": int(e.get("slice_ts_us", -1)),
                "width": int(m.get("width", -1)),
                "height": int(m.get("height", -1)),
                "mvs_wall_time_us": int(mvs_wall),
                "event_wall_time_us": int(event_wall),
                "wall_diff_us": int(mvs_wall - event_wall) if mvs_wall > 0 and event_wall > 0 else -999999999,
                "elapsed_since_grab_s": m.get("elapsed_since_grab_s", ""),
                "stable_count": int(m.get("stable_count", -1)),
                "gap_from_prev": int(m.get("gap_from_prev", -1)),
            })
            paired_counter += 1
            wrote += 1
        pending_rows[:] = remain
        if wrote > 0 and paired_file is not None:
            try:
                paired_file.flush()
            except Exception:
                pass
        return paired_counter

    device_list, devices = enum_devices()
    if args.list_devices:
        print("[MVS] detected devices:")
        for i, _dev, sn, model in devices:
            print(f"  [{i}] serial={sn} model={model}")
        return

    cam = MvCamera()
    base_writer = FfmpegSrtWriter(
        width=args.out_width,
        height=args.out_height,
        fps=args.fps,
        port=args.srt_port,
        bitrate_kbps=args.bitrate_kbps,
        latency_ms=args.latency,
        bind_ip=args.bind_ip,
        reconnect_delay_s=args.reconnect_delay,
        tlpktdrop=bool(args.tlpktdrop),
    )
    if bool(getattr(args, "async_srt", True)):
        writer = AsyncFfmpegSrtWriter(base_writer, queue_size=int(getattr(args, "async_queue_size", 4)))
    else:
        writer = base_writer

    obs_ts_align_enabled = bool(getattr(args, "obs_align_by_ts", False))
    if obs_ts_align_enabled:
        writer = TimestampAlignedFrameSender(
            writer,
            fps=float(args.fps),
            delay_ms=int(getattr(args, "obs_align_delay_ms", 1000)),
            buffer_frames=int(getattr(args, "obs_align_buffer_frames", 250)),
        )

    mvs_frame_sync_file = None
    mvs_frame_sync_writer = None
    if str(args.mvs_frame_sync_log or "").strip():
        log_path = str(args.mvs_frame_sync_log).strip()
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        mvs_frame_sync_file = open(log_path, "w", newline="", encoding="utf-8")
        mvs_frame_sync_writer = csv.DictWriter(mvs_frame_sync_file, fieldnames=[
            "frame_index", "mvs_frame_num", "width", "height", "wall_time_us",
        ])
        mvs_frame_sync_writer.writeheader()
        print(f"[MVS][SYNC] frame sync log enabled -> {log_path}", flush=True)

    sync_official_file = None
    sync_official_writer = None
    if str(args.sync_official_log or "").strip():
        official_path = str(args.sync_official_log).strip()
        official_dir = os.path.dirname(os.path.abspath(official_path))
        if official_dir:
            os.makedirs(official_dir, exist_ok=True)
        sync_official_file = open(official_path, "w", newline="", encoding="utf-8")
        sync_official_writer = csv.DictWriter(sync_official_file, fieldnames=[
            "official_pair_index",
            "mvs_frame_index", "mvs_frame_num", "expected_sync_index",
            "width", "height", "mvs_wall_time_us",
            "elapsed_since_grab_s", "stable_count", "gap_from_prev",
        ])
        sync_official_writer.writeheader()
        print(f"[MVS][SYNC] official stable sync log enabled -> {official_path}", flush=True)

    event_trigger_log_path = str(args.event_trigger_log or "").strip()
    sync_paired_file = None
    sync_paired_writer = None
    if str(args.sync_paired_official_log or "").strip():
        paired_path = str(args.sync_paired_official_log).strip()
        paired_dir = os.path.dirname(os.path.abspath(paired_path))
        if paired_dir:
            os.makedirs(paired_dir, exist_ok=True)
        sync_paired_file = open(paired_path, "w", newline="", encoding="utf-8")
        sync_paired_writer = csv.DictWriter(sync_paired_file, fieldnames=[
            "paired_index", "is_official", "sync_offset",
            "mvs_frame_index", "mvs_frame_num",
            "event_sync_index", "event_ts_us",
            "event_all_index", "event_polarity", "event_trigger_id", "event_slice_ts_us",
            "width", "height",
            "mvs_wall_time_us", "event_wall_time_us", "wall_diff_us",
            "elapsed_since_grab_s", "stable_count", "gap_from_prev",
        ])
        sync_paired_writer.writeheader()
        print(f"[MVS][SYNC] paired official sync log enabled -> {paired_path}", flush=True)
        if not event_trigger_log_path:
            print("[MVS][SYNC][WARN] --sync-paired-official-log 已启用，但没有提供 --event-trigger-log，无法写 event_ts_us", flush=True)
        else:
            print(f"[MVS][SYNC] will read event trigger log -> {event_trigger_log_path}", flush=True)

    # 这些变量在 finally 中也会用到；先初始化，避免启动早期异常时未定义。
    event_by_sync: Dict[int, Dict[str, int]] = {}
    pending_paired_rows: List[Dict[str, int]] = []
    paired_counter = 0
    sync_offset: Optional[int] = args.sync_fixed_offset
    sync_offset_locked = sync_offset is not None

    try:
        dev_info = select_device(device_list, serial=args.serial, index=args.index)

        ret = cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed, ret=0x{ret:x}")

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"MV_CC_OpenDevice failed, ret=0x{ret:x}。请确认 MVS 软件没有正在占用该相机。")

        set_if_possible(cam, "enum", "TriggerMode", MV_TRIGGER_MODE_OFF)

        if args.camera_fps is not None:
            set_if_possible(cam, "bool", "AcquisitionFrameRateEnable", True)
            set_if_possible(cam, "float", "AcquisitionFrameRate", args.camera_fps)

        if args.exposure_us is not None:
            set_if_possible(cam, "enum", "ExposureAuto", 0)
            set_if_possible(cam, "float", "ExposureTime", args.exposure_us)

        if args.gain is not None:
            set_if_possible(cam, "enum", "GainAuto", 0)
            set_if_possible(cam, "float", "Gain", args.gain)

        configure_mvs_sync_output(cam, args)

        # 关键：等待事件相机真正进入 External Trigger 记录循环后，再开始 MVS 采集。
        # 否则 MVS 可能先记录几十帧，导致 frame_index 和 sync_index 每次偏移不同。
        wait_ready_file_if_needed(args)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed, ret=0x{ret:x}")

        print(
            f"[MVS] streaming to OBS: "
            f"srt://采集机IP:{args.srt_port}?mode=caller&transtype=live&latency={int(args.latency) * 1000}&tlpktdrop={1 if args.tlpktdrop else 0}",
            flush=True,
        )
        print(f"[MVS] SRT auto reconnect delay = {args.reconnect_delay:.1f}s", flush=True)

        interval = 1.0 / max(0.1, args.fps)
        next_send = 0.0
        n = 0
        mvs_frame_index = 0
        official_pair_index = 0
        prev_mvs_frame_num = None
        stable_count = 0
        official_started = False
        t0 = time.time()
        grab_start_wall = t0
        warmup_s = max(0.0, float(getattr(args, "sync_warmup_seconds", 0.0) or 0.0))
        stable_min_frames = max(1, int(getattr(args, "sync_stable_min_frames", 1) or 1))
        gap_tolerance = max(0, int(getattr(args, "sync_stable_gap_tolerance", 0) or 0))

        # v5 自动配对状态。关系定义：event_sync_index = mvs_frame_num + sync_offset。
        event_by_sync: Dict[int, Dict[str, int]] = {}
        recent_mvs_official: List[Dict[str, int]] = []
        pending_paired_rows: List[Dict[str, int]] = []
        paired_counter = 0
        sync_offset: Optional[int] = args.sync_fixed_offset
        sync_offset_locked = sync_offset is not None
        last_event_reload_wall = 0.0
        offset_search_n = max(0, int(getattr(args, "sync_offset_search", 5) or 5))
        offset_min_pairs = max(1, int(getattr(args, "sync_offset_min_pairs", 20) or 20))
        offset_refresh_s = max(0.1, float(getattr(args, "sync_offset_refresh_s", 0.5) or 0.5))
        offset_max_wall_diff_us = int(max(0.0, float(getattr(args, "sync_offset_max_wall_diff_ms", 20.0) or 20.0)) * 1000.0)
        obs_align_reload_s = max(0.02, float(getattr(args, "obs_align_event_reload_s", 0.10) or 0.10))
        event_interval_us = int(round(1_000_000.0 / max(0.1, float(args.fps))))
        obs_align_exact_count = 0
        obs_align_est_count = 0
        obs_align_missing_count = 0
        obs_align_last_report_wall = 0.0

        print(
            f"[MVS][SYNC] warmup policy: warmup_s={warmup_s:.3f}, "
            f"stable_min_frames={stable_min_frames}, gap_tolerance={gap_tolerance}",
            flush=True,
        )
        if sync_paired_writer is not None:
            if sync_offset_locked:
                print(f"[MVS][SYNC] fixed offset: event_sync_index = mvs_frame_num + ({sync_offset})", flush=True)
            else:
                print(
                    f"[MVS][SYNC] auto offset enabled: search=[-{offset_search_n}, +{offset_search_n}], "
                    f"min_pairs={offset_min_pairs}, max_median_abs_wall_diff={offset_max_wall_diff_us/1000.0:.3f}ms",
                    flush=True,
                )

        while True:
            frame_out = MV_FRAME_OUT()
            memset(byref(frame_out), 0, sizeof(frame_out))

            ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
            if ret != 0:
                print(f"[MVS][WARN] GetImageBuffer timeout/failed ret=0x{ret:x}")
                continue

            try:
                now = time.time()
                wall_time_us = int(now * 1e6)
                info = frame_out.stFrameInfo
                try:
                    mvs_frame_num = int(info.nFrameNum)
                except Exception:
                    mvs_frame_num = -1
                width_i = int(getattr(info, "nWidth", 0))
                height_i = int(getattr(info, "nHeight", 0))

                gap_from_prev = 0
                if prev_mvs_frame_num is not None and mvs_frame_num >= 0 and prev_mvs_frame_num >= 0:
                    gap_from_prev = int(mvs_frame_num) - int(prev_mvs_frame_num) - 1
                    if gap_from_prev < 0:
                        gap_from_prev = 0
                gap_ok = (prev_mvs_frame_num is None) or (gap_from_prev <= gap_tolerance)
                stable_count = stable_count + 1 if gap_ok else 1
                if (not gap_ok) and args.print_every >= 0:
                    print(
                        f"[MVS][SYNC][GAP] frame_num jump {prev_mvs_frame_num} -> {mvs_frame_num}, "
                        f"missing={gap_from_prev}",
                        flush=True,
                    )
                prev_mvs_frame_num = mvs_frame_num

                if mvs_frame_sync_writer is not None:
                    mvs_frame_sync_writer.writerow({
                        "frame_index": int(mvs_frame_index),
                        "mvs_frame_num": int(mvs_frame_num),
                        "width": width_i,
                        "height": height_i,
                        "wall_time_us": wall_time_us,
                    })
                    if mvs_frame_index % 100 == 0:
                        mvs_frame_sync_file.flush()

                elapsed_since_grab = now - grab_start_wall
                if not official_started:
                    if elapsed_since_grab >= warmup_s and stable_count >= stable_min_frames:
                        official_started = True
                        print(
                            f"[MVS][SYNC] OFFICIAL_START "
                            f"mvs_frame_index={mvs_frame_index} mvs_frame_num={mvs_frame_num} "
                            f"elapsed={elapsed_since_grab:.3f}s stable_count={stable_count}",
                            flush=True,
                        )
                    else:
                        # 预热/未稳定阶段：只快速取帧并释放，避免 ffmpeg/SRT/显示阻塞 MVS 取帧。
                        if args.print_every > 0 and mvs_frame_index % args.print_every == 0:
                            print(
                                f"[MVS][SYNC] warming up: frame_index={mvs_frame_index} "
                                f"frame_num={mvs_frame_num} elapsed={elapsed_since_grab:.2f}s "
                                f"stable_count={stable_count}/{stable_min_frames}",
                                flush=True,
                            )
                        mvs_frame_index += 1
                        continue

                current_event_sync_index = int(mvs_frame_num) + int(sync_offset) if (sync_offset is not None and mvs_frame_num >= 0) else -1
                official_row = {
                    "official_pair_index": int(official_pair_index),
                    "mvs_frame_index": int(mvs_frame_index),
                    "mvs_frame_num": int(mvs_frame_num),
                    "expected_sync_index": int(current_event_sync_index),
                    "width": int(width_i),
                    "height": int(height_i),
                    "mvs_wall_time_us": int(wall_time_us),
                    "elapsed_since_grab_s": f"{elapsed_since_grab:.6f}",
                    "stable_count": int(stable_count),
                    "gap_from_prev": int(gap_from_prev),
                }

                if sync_official_writer is not None:
                    sync_official_writer.writerow(official_row)
                    if official_pair_index % 100 == 0:
                        sync_official_file.flush()

                # v5：把 official MVS 帧放进 pending，等 event trigger CSV 刷新后自动补齐 event_ts_us。
                if sync_paired_writer is not None and mvs_frame_num >= 0:
                    pending_paired_rows.append({
                        "mvs_frame_index": int(mvs_frame_index),
                        "mvs_frame_num": int(mvs_frame_num),
                        "width": int(width_i),
                        "height": int(height_i),
                        "mvs_wall_time_us": int(wall_time_us),
                        "elapsed_since_grab_s": f"{elapsed_since_grab:.6f}",
                        "stable_count": int(stable_count),
                        "gap_from_prev": int(gap_from_prev),
                    })
                    recent_mvs_official.append({
                        "mvs_frame_num": int(mvs_frame_num),
                        "mvs_wall_time_us": int(wall_time_us),
                    })
                    # recent 只用于 offset 估计，保留最近 500 帧足够。
                    if len(recent_mvs_official) > 500:
                        del recent_mvs_official[:-500]

                    reload_s = obs_align_reload_s if obs_ts_align_enabled else offset_refresh_s
                    if event_trigger_log_path and (now - last_event_reload_wall >= reload_s):
                        event_by_sync = load_event_trigger_rows(event_trigger_log_path)
                        event_interval_us = estimate_event_interval_us(event_by_sync, event_interval_us)
                        last_event_reload_wall = now

                        if not sync_offset_locked:
                            est = estimate_sync_offset(
                                recent_mvs_official,
                                event_by_sync,
                                offset_search_n,
                                offset_min_pairs,
                                offset_max_wall_diff_us,
                            )
                            if est is not None:
                                sync_offset, pairs, med_abs, med_signed = est
                                sync_offset_locked = True
                                print(
                                    f"[MVS][SYNC] AUTO_OFFSET_LOCKED offset={sync_offset} "
                                    f"relation: event_sync_index=mvs_frame_num+({sync_offset}), "
                                    f"pairs={pairs}, median_abs_wall_diff_us={med_abs:.1f}, "
                                    f"median_signed_wall_diff_us={med_signed:.1f}",
                                    flush=True,
                                )

                        paired_counter = try_write_paired_rows(
                            pending_paired_rows,
                            event_by_sync,
                            sync_paired_writer,
                            sync_paired_file,
                            sync_offset if sync_offset_locked else None,
                            paired_counter,
                        )

                official_pair_index += 1
                mvs_frame_index += 1

                if now < next_send:
                    continue

                bgr = convert_frame_to_bgr(cam, frame_out)
                out = resize_fit(bgr, args.out_width, args.out_height, args.fit)

                if obs_ts_align_enabled:
                    mapped_event_ts_us = -1
                    mapped_kind = "not_locked"
                    if sync_offset_locked and sync_offset is not None and mvs_frame_num >= 0:
                        event_sync_index_for_obs = int(mvs_frame_num) + int(sync_offset)
                        mapped_event_ts_us, mapped_kind = get_event_ts_for_sync_index(
                            event_by_sync,
                            event_sync_index_for_obs,
                            event_interval_us,
                        )
                    if mapped_event_ts_us >= 0 and hasattr(writer, "submit"):
                        writer.submit(mapped_event_ts_us, out)
                        if mapped_kind == "exact":
                            obs_align_exact_count += 1
                        else:
                            obs_align_est_count += 1
                    else:
                        obs_align_missing_count += 1

                    if now - obs_align_last_report_wall >= 5.0:
                        print(
                            f"[MVS][OBS-TS-ALIGN] map exact={obs_align_exact_count} "
                            f"estimated={obs_align_est_count} missing={obs_align_missing_count} "
                            f"offset_locked={int(sync_offset_locked)} offset={sync_offset} "
                            f"event_interval_us={event_interval_us}",
                            flush=True,
                        )
                        obs_align_last_report_wall = now
                else:
                    if not writer.write(out):
                        time.sleep(0.05)

                n += 1
                next_send = now + interval

                if args.show_local:
                    cv2.imshow("hik_mvs_obs_srt", out)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if args.print_every > 0 and n % args.print_every == 0:
                    dt = max(1e-6, time.time() - t0)
                    print(f"[MVS] sent={n}, avg_fps={n/dt:.1f}", flush=True)

            finally:
                cam.MV_CC_FreeImageBuffer(frame_out)

    except KeyboardInterrupt:
        print("\n[MVS] interrupted")
    finally:
        try:
            writer.close()
        except Exception:
            pass
        try:
            cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        if mvs_frame_sync_file is not None:
            try:
                mvs_frame_sync_file.close()
            except Exception:
                pass
        if sync_official_file is not None:
            try:
                sync_official_file.close()
            except Exception:
                pass
        if sync_paired_file is not None:
            try:
                # 结束前最后再尝试读一次 event log，补写已经 flush 的 pending 配对。
                if event_trigger_log_path:
                    event_by_sync = load_event_trigger_rows(event_trigger_log_path)
                    paired_counter = try_write_paired_rows(
                        pending_paired_rows,
                        event_by_sync,
                        sync_paired_writer,
                        sync_paired_file,
                        sync_offset if sync_offset_locked else None,
                        paired_counter,
                    )
                sync_paired_file.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[MVS] closed")


if __name__ == "__main__":
    main()
