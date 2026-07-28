"""
obs_srt_streamer.py — 直接把 OpenCV BGR 画面异步推送为 OBS 可接收的 SRT 视频流

设计目标：
  - 主循环只调用 submit(frame)，不做网络 I/O，不阻塞子弹检测。
  - 内部只保留最新帧，旧帧自动丢弃，避免 OBS 或网络卡顿造成积压。
  - 使用 ffmpeg 从 stdin 接收 raw BGR24，编码为 H.264，再通过 SRT listener 输出。
  - OBS 端使用 Media Source，取消 Local File，填：
      srt://采集机IP:端口?mode=caller&transtype=live&latency=80000&tlpktdrop=1
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


# LOW_LATENCY_SRT_PATCH: convert latency_ms to microseconds for ffmpeg/libsrt URLs.
class ObsSrtStreamer:
    """
    OpenCV BGR 帧 -> ffmpeg stdin -> H.264/MPEG-TS/SRT listener。

    submit() 是主循环调用入口：
      - 未到发送帧率上限时直接返回；
      - 到点后复制一份最新帧交给后台线程；
      - 后台线程负责 resize、裁剪偶数尺寸、写入 ffmpeg。
    """

    def __init__(
        self,
        port: int,
        enabled: bool = True,
        fps: float = 25.0,
        bitrate_kbps: int = 2500,
        latency_ms: int = 80,
        scale: float = 1.0,
        bind_ip: str = "0.0.0.0",
        ffmpeg_bin: str = "ffmpeg",
        encoder: str = "libx264",
        preset: str = "ultrafast",
        gop: int = 0,
        label: str = "",
        log_file: str = "",
        tlpktdrop: bool = False,
        align_by_ts: bool = False,
        align_delay_ms: int = 1000,
        align_buffer_frames: int = 250,
        max_consecutive_failures: int = 8,
        circuit_breaker_sec: float = 60.0,
        max_backoff_sec: float = 30.0,
    ) -> None:
        self.enabled = bool(enabled) and int(port) > 0
        self.port = int(port)
        self.fps = max(0.1, float(fps))
        self.bitrate_kbps = int(bitrate_kbps)
        self.latency_ms = int(latency_ms)
        self.scale = max(0.05, float(scale))
        self.bind_ip = str(bind_ip or "0.0.0.0")
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.encoder = str(encoder or "libx264")
        self.preset = str(preset or "ultrafast")
        self.gop = int(gop) if int(gop or 0) > 0 else int(round(self.fps))
        self.label = str(label or f"port{self.port}")
        self.log_file = str(log_file or "")
        # OBS 完整性优先：默认不丢迟到包。低延迟优先时可显式开启 tlpktdrop。
        self.tlpktdrop = bool(tlpktdrop)

        # v3: 按事件相机 sensor timestamp 对齐 OBS 输出。
        # 普通模式 submit(frame) 仍然只保留最新帧；
        # 对齐模式 submit_at_ts(frame, ts_us) 会先进入 timestamp buffer，
        # 后台线程按 target_ts = latest_ts - align_delay_us 输出最接近的历史帧。
        self.align_by_ts = bool(align_by_ts)
        self.align_delay_us = max(0, int(align_delay_ms) * 1000)
        self.align_buffer_frames = max(10, int(align_buffer_frames))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures or 8))
        self.circuit_breaker_sec = max(1.0, float(circuit_breaker_sec or 60.0))
        self.max_backoff_sec = max(0.25, float(max_backoff_sec or 30.0))
        self.restart_storm_window_sec = 30.0
        self._ts_buffer: list[tuple[int, np.ndarray]] = []
        self._latest_ts_us: Optional[int] = None
        self._align_submitted = 0
        self._align_output = 0
        self._align_wait_no_frame = 0

        self._cond = threading.Condition()
        self._pending: Optional[np.ndarray] = None
        self._stop = False
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"ObsSrtStreamer-{self.label}-{self.port}",
        )
        self._started = False
        self._proc: Optional[subprocess.Popen] = None
        self._last_submit_wall = 0.0
        self._last_err_wall = 0.0
        self._last_start_wall = 0.0
        self._restart_count = 0
        self._consecutive_failures = 0
        self._restart_backoff_s = 0.25
        self._restart_backoff_until = 0.0
        self._circuit_open_until = 0.0
        self._circuit_open_count = 0
        self._last_restart_reason = ""
        self._failure_wall_times: list[float] = []
        self._last_exit_rc: Optional[int] = None
        self._log_fh = None
        self._log_path: Optional[Path] = None
        self._log_initialized = False
        self._last_stderr_tail = ""
        self._last_bind_error = ""
        self._sent_frames = 0
        self._drop_frames = 0

    @property
    def url_for_obs(self) -> str:
        latency_us = max(1, int(self.latency_ms) * 1000)
        return (
            f"srt://<采集机IP>:{self.port}"
            f"?mode=caller&transtype=live&latency={latency_us}&tlpktdrop={1 if self.tlpktdrop else 0}"
        )

    def start(self) -> None:
        if not self.enabled:
            print("[OBS-SRT] disabled")
            return
        if self._started:
            return
        self._started = True
        self._thread.start()
        print(
            f"[OBS-SRT][{self.label}] enabled: "
            f"listen={self.bind_ip}:{self.port}, fps={self.fps}, "
            f"bitrate={self.bitrate_kbps}k, obs_url={self.url_for_obs}",
            flush=True,
        )
        if self.align_by_ts:
            print(
                f"[OBS-SRT][{self.label}][TS-ALIGN] enabled: "
                f"delay_ms={self.align_delay_us/1000.0:.1f}, buffer_frames={self.align_buffer_frames}",
                flush=True,
            )

    def submit(self, frame: np.ndarray) -> bool:
        """
        提交一帧 BGR/RGB 图像。主循环安全调用。
        返回 True 表示接受了这一帧，False 表示被帧率限制或未启用。
        """
        if not self.enabled or frame is None:
            return False

        now = time.time()
        min_interval = 1.0 / self.fps
        if now - self._last_submit_wall < min_interval:
            self._drop_frames += 1
            return False
        self._last_submit_wall = now

        try:
            # output_img 在主循环里会被复用，必须复制一份交给后台线程。
            frm = frame.copy()
        except Exception:
            return False

        with self._cond:
            # 只保留最新帧，丢弃旧帧，避免延迟积压。
            if self._pending is not None:
                self._drop_frames += 1
            self._pending = frm
            self._cond.notify()
        return True

    def submit_at_ts(self, frame: np.ndarray, ts_us: int) -> bool:
        """提交带 sensor timestamp 的帧。

        未启用 align_by_ts 时退化为普通 submit(frame)。启用后，后台线程会
        按 target_ts = latest_ts_us - align_delay_us 从 buffer 选择历史帧输出，
        用于让事件画面和 MVS 画面在 OBS 端使用同一个 event_ts_us 时间轴。
        """
        if not self.align_by_ts:
            return self.submit(frame)
        if not self.enabled or frame is None:
            return False
        try:
            ts_i = int(ts_us)
        except Exception:
            return False
        if ts_i < 0:
            return False
        try:
            frm = frame.copy()
        except Exception:
            return False

        with self._cond:
            # 大多数情况下 ts 单调递增，直接 append；如果偶发乱序，做一次插入保证排序。
            if not self._ts_buffer or ts_i >= self._ts_buffer[-1][0]:
                self._ts_buffer.append((ts_i, frm))
            else:
                inserted = False
                for i, (old_ts, _old_frame) in enumerate(self._ts_buffer):
                    if ts_i < old_ts:
                        self._ts_buffer.insert(i, (ts_i, frm))
                        inserted = True
                        break
                if not inserted:
                    self._ts_buffer.append((ts_i, frm))
            self._latest_ts_us = ts_i if self._latest_ts_us is None else max(self._latest_ts_us, ts_i)
            while len(self._ts_buffer) > self.align_buffer_frames:
                self._ts_buffer.pop(0)
                self._drop_frames += 1
            self._align_submitted += 1
            self._cond.notify()
        return True

    def close(self) -> None:
        if not self.enabled:
            return
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        # 先停 ffmpeg，避免后台线程正阻塞在 stdin.write 时 close 等不到。
        self._stop_ffmpeg()

        if self._started:
            self._thread.join(timeout=2.0)

        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

        print(
            f"[OBS-SRT][{self.label}] closed. sent={self._sent_frames} dropped={self._drop_frames} "
            f"align_submitted={self._align_submitted} align_output={self._align_output}",
            flush=True,
        )

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        proc = self._proc
        proc_alive = bool(proc is not None and proc.poll() is None)
        with self._cond:
            pending = self._pending is not None
            ts_buffer_len = len(self._ts_buffer)
            latest_ts_us = int(self._latest_ts_us or 0)
        restart_remaining = max(0.0, float(self._restart_backoff_until) - now)
        circuit_remaining = max(0.0, float(self._circuit_open_until) - now)
        return {
            "enabled": bool(self.enabled),
            "label": str(self.label),
            "port": int(self.port),
            "started": bool(self._started),
            "proc_alive": bool(proc_alive),
            "fps": float(self.fps),
            "sent_frames": int(self._sent_frames),
            "drop_frames": int(self._drop_frames),
            "restart_count": int(self._restart_count),
            "consecutive_failures": int(self._consecutive_failures),
            "failure_window_count": int(len(self._failure_wall_times)),
            "failure_window_sec": round(float(self.restart_storm_window_sec), 3),
            "max_consecutive_failures": int(self.max_consecutive_failures),
            "restart_backoff_s": round(float(self._restart_backoff_s), 3),
            "restart_backoff_remaining_s": round(float(restart_remaining), 3),
            "circuit_open": bool(circuit_remaining > 0.0),
            "circuit_open_remaining_s": round(float(circuit_remaining), 3),
            "circuit_open_count": int(self._circuit_open_count),
            "circuit_breaker_sec": round(float(self.circuit_breaker_sec), 3),
            "last_exit_rc": self._last_exit_rc,
            "last_restart_reason": str(self._last_restart_reason),
            "ffmpeg_log_path": str(self._log_path or ""),
            "last_ffmpeg_stderr_tail": str(self._last_stderr_tail),
            "last_bind_error": str(self._last_bind_error),
            "pending_frame": bool(pending),
            "align_by_ts": bool(self.align_by_ts),
            "align_submitted": int(self._align_submitted),
            "align_output": int(self._align_output),
            "align_wait_no_frame": int(self._align_wait_no_frame),
            "align_buffer_frames": int(self.align_buffer_frames),
            "align_buffer_len": int(ts_buffer_len),
            "latest_ts_us": int(latest_ts_us),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        else:
            raise ValueError(f"unsupported frame shape: {frame.shape}")

        if abs(self.scale - 1.0) > 1e-6:
            h, w = frame.shape[:2]
            nw = max(2, int(round(w * self.scale)))
            nh = max(2, int(round(h * self.scale)))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        # yuv420p / H.264 对偶数宽高更稳，直接裁掉最后一行/列。
        h, w = frame.shape[:2]
        even_w = w // 2 * 2
        even_h = h // 2 * 2
        if even_w != w or even_h != h:
            frame = frame[:even_h, :even_w, :]

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        return frame

    def _build_ffmpeg_cmd(self, w: int, h: int) -> list[str]:
        # ffmpeg/libsrt URL 中 latency 的单位是微秒；外部参数 latency_ms 仍按毫秒传入。
        latency_us = max(1, int(self.latency_ms) * 1000)
        srt_url = (
            f"srt://{self.bind_ip}:{self.port}"
            f"?mode=listener"
            f"&transtype=live"
            f"&latency={latency_us}"
            f"&tlpktdrop={1 if self.tlpktdrop else 0}"
            f"&pkt_size=1316"
        )

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{w}x{h}",
            "-framerate", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", self.encoder,
        ]

        if self.encoder == "libx264":
            cmd += [
                "-preset", self.preset,
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-threads", "2",
                "-x264-params", f"keyint={max(1, self.gop)}:min-keyint={max(1, self.gop)}:scenecut=0",
            ]
        elif "nvenc" in self.encoder:
            # 如果用户显式选择 h264_nvenc，可减少 CPU，但不同机器 preset 支持不同，保持参数保守。
            cmd += ["-pix_fmt", "yuv420p"]

        cmd += [
            "-b:v", f"{self.bitrate_kbps}k",
            "-maxrate", f"{self.bitrate_kbps}k",
            "-bufsize", f"{max(500, self.bitrate_kbps // 2)}k",
            "-g", str(max(1, self.gop)),
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "mpegts",
            srt_url,
        ]
        return cmd

    def _probe_udp_bind_available(self) -> tuple[bool, str]:
        if int(self.port) <= 0:
            return False, "invalid_port"
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.bind_ip, int(self.port)))
            return True, ""
        except OSError as exc:
            return False, f"{self.bind_ip}:{self.port} unavailable before ffmpeg start: {exc}"
        except Exception as exc:
            return False, f"{self.bind_ip}:{self.port} probe failed: {exc}"
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _refresh_ffmpeg_log_tail(self, max_bytes: int = 4096) -> None:
        path = self._log_path
        if path is None:
            return
        try:
            if self._log_fh is not None and hasattr(self._log_fh, "flush"):
                self._log_fh.flush()
        except Exception:
            pass
        try:
            if not path.exists():
                return
            with open(path, "rb") as fh:
                try:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    fh.seek(max(0, size - int(max_bytes)), os.SEEK_SET)
                except Exception:
                    pass
                data = fh.read()
            text = data.decode("utf-8", errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        self._last_stderr_tail = text[-2000:]
        lowered = text.lower()
        if "address already in use" in lowered or "unable to create/configure srt socket" in lowered:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            self._last_bind_error = lines[-1][-500:] if lines else "ffmpeg bind error"

    def _start_ffmpeg(self, w: int, h: int) -> None:
        self._stop_ffmpeg()

        if self.log_file:
            log_path = Path(self.log_file)
        else:
            safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in self.label)
            log_path = Path.cwd() / f"obs_srt_{safe_label}_{self.port}.log"
        self._log_path = log_path

        ok, bind_error = self._probe_udp_bind_available()
        if not ok:
            self._last_bind_error = bind_error
            self._mark_ffmpeg_restart_delay(f"port unavailable: {bind_error}")
            return

        try:
            mode = "a" if self._log_initialized else "w"
            self._log_fh = open(log_path, mode, buffering=1, encoding="utf-8", errors="replace")
            self._log_initialized = True
        except Exception:
            self._log_fh = subprocess.DEVNULL

        cmd = self._build_ffmpeg_cmd(w, h)
        print(f"[OBS-SRT][{self.label}] start ffmpeg: {' '.join(cmd)}", flush=True)
        self._last_start_wall = time.time()

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_fh,
            bufsize=0,
        )

    def _stop_ffmpeg(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._refresh_ffmpeg_log_tail()
        if self._log_fh is not None and hasattr(self._log_fh, "close"):
            try:
                self._log_fh.close()
            except Exception:
                pass
        self._log_fh = None

    def _rate_limited_error(self, msg: str) -> None:
        now = time.time()
        if now - self._last_err_wall > 1.0:
            print(f"[OBS-SRT][{self.label}] {msg}", flush=True)
            self._last_err_wall = now

    def _mark_ffmpeg_restart_delay(self, reason: str) -> None:
        now = time.time()
        self._refresh_ffmpeg_log_tail()
        self._restart_count += 1
        self._consecutive_failures += 1
        self._last_restart_reason = str(reason)
        self._failure_wall_times.append(float(now))
        cutoff = now - float(self.restart_storm_window_sec)
        self._failure_wall_times = [t for t in self._failure_wall_times if float(t) >= cutoff]
        if now - self._last_start_wall < 5.0:
            self._restart_backoff_s = min(self.max_backoff_sec, max(0.25, self._restart_backoff_s * 1.7))
        else:
            self._restart_backoff_s = max(0.25, self._restart_backoff_s * 0.5)
        self._restart_backoff_until = now + self._restart_backoff_s
        restart_storm = bool(len(self._failure_wall_times) >= self.max_consecutive_failures)
        if self._consecutive_failures >= self.max_consecutive_failures or restart_storm:
            self._circuit_open_count += 1
            self._circuit_open_until = max(self._circuit_open_until, now + self.circuit_breaker_sec)
            self._restart_backoff_until = self._circuit_open_until
            self._restart_backoff_s = min(self.max_backoff_sec, max(self._restart_backoff_s, self.circuit_breaker_sec))
        self._rate_limited_error(
            f"ffmpeg unavailable ({reason}); consecutive_failures={self._consecutive_failures} "
            f"failure_window={len(self._failure_wall_times)}/{self.max_consecutive_failures} "
            f"circuit_open={now < self._circuit_open_until} restart_backoff={self._restart_backoff_s:.2f}s"
        )

    def _write_frame_to_ffmpeg(self, frame: np.ndarray, last_size):
        try:
            frame = self._prepare_frame(frame)
            h, w = frame.shape[:2]
            cur_size = (w, h)

            proc = self._proc
            if proc is not None:
                rc = proc.poll()
                if rc is not None:
                    self._last_exit_rc = int(rc)
                    self._proc = None
                    self._mark_ffmpeg_restart_delay(f"exited rc={rc}")

            if cur_size != last_size and self._proc is not None:
                self._stop_ffmpeg()

            if self._proc is None:
                now = time.time()
                if now < self._circuit_open_until:
                    return cur_size
                if now < self._restart_backoff_until:
                    return cur_size
                last_size = cur_size
                self._start_ffmpeg(w, h)

            proc = self._proc
            if proc is None or proc.stdin is None:
                return last_size

            # 若 OBS 尚未连接，ffmpeg 可能在 SRT listener 打开阶段等待；
            # 即使这里阻塞，也只阻塞后台线程，不影响主检测循环。
            proc.stdin.write(frame.tobytes())
            self._sent_frames += 1
            if self._restart_count:
                self._restart_count = 0
                self._consecutive_failures = 0
                self._restart_backoff_s = 0.25
                self._restart_backoff_until = 0.0
            return last_size

        except (BrokenPipeError, OSError) as exc:
            self._stop_ffmpeg()
            self._mark_ffmpeg_restart_delay(f"pipe broken: {exc}")
            return None
        except Exception as exc:
            self._rate_limited_error(f"worker error: {exc}")
            time.sleep(0.05)
            return last_size

    def _pop_aligned_frame_locked(self) -> Optional[np.ndarray]:
        if not self._ts_buffer or self._latest_ts_us is None:
            return None
        target_ts = int(self._latest_ts_us) - int(self.align_delay_us)
        # 缓冲还没覆盖到 delay，先等待，不提前输出未来帧。
        if self._ts_buffer[0][0] > target_ts:
            return None

        idx = None
        for i, (ts_i, _frame) in enumerate(self._ts_buffer):
            if ts_i <= target_ts:
                idx = i
            else:
                break
        if idx is None:
            return None
        ts_i, frame = self._ts_buffer[idx]
        # 丢掉已经输出及更旧帧；如果 OBS 输出慢，会自动跳到最新可对齐帧。
        del self._ts_buffer[:idx + 1]
        self._align_output += 1
        return frame

    def _worker(self) -> None:
        last_size = None
        interval = 1.0 / max(0.1, float(self.fps))

        if self.align_by_ts:
            next_tick = time.time()
            while True:
                now = time.time()
                wait_s = max(0.0, next_tick - now)
                with self._cond:
                    if wait_s > 0 and not self._stop:
                        self._cond.wait(timeout=min(wait_s, 0.2))
                    if self._stop:
                        break
                    frame = self._pop_aligned_frame_locked()

                if frame is None:
                    self._align_wait_no_frame += 1
                    # 等新 timestamp 进入；不要忙等。
                    with self._cond:
                        if not self._stop:
                            self._cond.wait(timeout=0.02)
                    next_tick = max(next_tick, time.time())
                    continue

                last_size = self._write_frame_to_ffmpeg(frame, last_size)
                next_tick += interval
                # 如果编码/SRT 写入慢导致落后，直接重置节拍，避免补发造成积压。
                if next_tick < time.time() - interval:
                    next_tick = time.time() + interval

            self._stop_ffmpeg()
            return

        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.2)
                if self._stop:
                    break
                frame = self._pending
                self._pending = None

            if frame is None:
                continue

            last_size = self._write_frame_to_ffmpeg(frame, last_size)

        self._stop_ffmpeg()
