from __future__ import annotations

import atexit
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np


EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
TRIGGER_DTYPE = np.dtype([("p", "<i2"), ("id", "<i2"), ("t", "<i8")])

_HEADER_STRUCT = struct.Struct("<4sHHIIIQqqqIIIIII")
_MAGIC = b"EVB1"
_FLAG_HELLO = 1 << 0
_FLAG_FINAL = 1 << 1


class HalEventPipeIterator:
    """EventsIterator-compatible wrapper around the C++ HAL pipe bridge."""

    def __init__(
        self,
        *,
        bridge_bin: str,
        serial: str,
        slice_us: int,
        wait_mode: str = "poll",
        poll_sleep_us: int = 100,
        enable_trigger_in: bool = False,
        enable_erc: bool = False,
        erc_rate: Optional[int] = None,
        enable_trail: bool = False,
        trail_type: str = "stc_keep_trail",
        trail_threshold_us: int = 10000,
        biases: Optional[List[str]] = None,
        stderr_log: str = "",
    ):
        self.bridge_bin = str(bridge_bin)
        self.serial = str(serial or "")
        self.slice_us = int(slice_us or 4000)
        self.wait_mode = str(wait_mode or "poll")
        self.poll_sleep_us = int(poll_sleep_us or 0)
        self._current_time = 0
        self._width = 0
        self._height = 0
        self._closed = False
        self._pending_triggers = []
        self._stderr_file = None
        self._stderr_thread = None

        cmd = [
            self.bridge_bin,
            "--serial", self.serial,
            "--slice-us", str(self.slice_us),
            "--wait-mode", self.wait_mode,
            "--poll-sleep-us", str(self.poll_sleep_us),
        ]
        if enable_trigger_in:
            cmd.append("--enable-trigger-in")
        if enable_erc:
            cmd.append("--enable-erc")
            if erc_rate:
                cmd.extend(["--erc-rate", str(int(erc_rate))])
        if enable_trail:
            cmd.extend([
                "--enable-trail",
                "--trail-type", str(trail_type or "stc_keep_trail"),
                "--trail-threshold-us", str(int(trail_threshold_us or 10000)),
            ])
        for entry in biases or []:
            if entry:
                cmd.extend(["--set-bias", str(entry)])

        stderr = subprocess.PIPE
        log_path = str(stderr_log or "").strip()
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = open(log_path, "ab", buffering=0)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
        atexit.register(self.close)
        self._start_stderr_drain()
        self._read_hello()

    def _start_stderr_drain(self):
        def _drain():
            if self.proc.stderr is None:
                return
            for chunk in iter(lambda: self.proc.stderr.readline(), b""):
                if not chunk:
                    break
                if self._stderr_file is not None:
                    try:
                        self._stderr_file.write(chunk)
                    except Exception:
                        pass
                else:
                    try:
                        sys.stderr.buffer.write(chunk)
                        sys.stderr.buffer.flush()
                    except Exception:
                        pass

        self._stderr_thread = threading.Thread(target=_drain, name="HalEventPipeStderr", daemon=True)
        self._stderr_thread.start()

    def _read_exact(self, n: int) -> bytes:
        if self.proc.stdout is None:
            raise StopIteration
        chunks = []
        remain = int(n)
        while remain > 0:
            chunk = self.proc.stdout.read(remain)
            if not chunk:
                rc = self.proc.poll()
                if rc not in (None, 0):
                    raise RuntimeError(f"HAL bridge exited with rc={rc}")
                raise StopIteration
            chunks.append(chunk)
            remain -= len(chunk)
        return b"".join(chunks)

    def _read_frame(self):
        header_bytes = self._read_exact(_HEADER_STRUCT.size)
        fields = _HEADER_STRUCT.unpack(header_bytes)
        (
            magic, version, header_size, flags, width, height, seq,
            slice_start_us, slice_end_us, current_time_us,
            event_count, trigger_count, event_record_size, trigger_record_size,
            _payload_crc, _reserved,
        ) = fields
        if magic != _MAGIC:
            raise RuntimeError(f"bad HAL bridge magic: {magic!r}")
        if version != 1:
            raise RuntimeError(f"unsupported HAL bridge version: {version}")
        if header_size > _HEADER_STRUCT.size:
            self._read_exact(header_size - _HEADER_STRUCT.size)
        if event_record_size != EVENT_DTYPE.itemsize:
            raise RuntimeError(f"unexpected event record size: {event_record_size}")
        if trigger_record_size != TRIGGER_DTYPE.itemsize:
            raise RuntimeError(f"unexpected trigger record size: {trigger_record_size}")

        event_payload = self._read_exact(int(event_count) * EVENT_DTYPE.itemsize)
        trigger_payload = self._read_exact(int(trigger_count) * TRIGGER_DTYPE.itemsize)
        events = np.frombuffer(event_payload, dtype=EVENT_DTYPE, count=int(event_count))
        triggers = np.frombuffer(trigger_payload, dtype=TRIGGER_DTYPE, count=int(trigger_count))
        return {
            "flags": int(flags),
            "width": int(width),
            "height": int(height),
            "seq": int(seq),
            "slice_start_us": int(slice_start_us),
            "slice_end_us": int(slice_end_us),
            "current_time_us": int(current_time_us),
            "events": events,
            "triggers": triggers,
        }

    def _read_hello(self):
        while True:
            frame = self._read_frame()
            self._width = int(frame["width"])
            self._height = int(frame["height"])
            if frame["flags"] & _FLAG_HELLO:
                return
            self._pending_first_frame = frame
            return

    def __iter__(self):
        return self

    def __next__(self):
        frame = getattr(self, "_pending_first_frame", None)
        if frame is not None:
            self._pending_first_frame = None
        else:
            while True:
                frame = self._read_frame()
                if frame["flags"] & _FLAG_HELLO:
                    self._width = int(frame["width"])
                    self._height = int(frame["height"])
                    continue
                break
        if frame["flags"] & _FLAG_FINAL:
            raise StopIteration
        self._current_time = int(frame["current_time_us"])
        triggers = frame["triggers"]
        if len(triggers) > 0:
            self._pending_triggers.append(triggers)
        return frame["events"]

    def get_current_time(self):
        return int(self._current_time)

    def get_size(self):
        return int(self._height), int(self._width)

    def get_ext_trigger_events(self):
        if not self._pending_triggers:
            return []
        if len(self._pending_triggers) == 1:
            return self._pending_triggers[0]
        return np.concatenate(self._pending_triggers)

    def clear_ext_trigger_events(self):
        self._pending_triggers.clear()

    def close(self):
        if self._closed:
            return
        self._closed = True
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass


class HalEventFifoIterator(HalEventPipeIterator):
    """EventsIterator-compatible reader for a FIFO written by hal_event_fifo_broker."""

    def __init__(self, *, fifo_path: str):
        self.bridge_bin = ""
        self.serial = ""
        self.slice_us = 0
        self.wait_mode = "fifo"
        self.poll_sleep_us = 0
        self._current_time = 0
        self._width = 0
        self._height = 0
        self._closed = False
        self._pending_triggers = []
        self._stderr_file = None
        self._stderr_thread = None
        self.proc = None
        self.fifo_path = str(fifo_path)
        self._fifo_file = open(self.fifo_path, "rb", buffering=0)
        atexit.register(self.close)
        self._read_hello()

    def _start_stderr_drain(self):
        return

    def _read_exact(self, n: int) -> bytes:
        f = getattr(self, "_fifo_file", None)
        if f is None:
            raise StopIteration
        chunks = []
        remain = int(n)
        while remain > 0:
            chunk = f.read(remain)
            if not chunk:
                raise StopIteration
            chunks.append(chunk)
            remain -= len(chunk)
        return b"".join(chunks)

    def close(self):
        if self._closed:
            return
        self._closed = True
        f = getattr(self, "_fifo_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._fifo_file = None
