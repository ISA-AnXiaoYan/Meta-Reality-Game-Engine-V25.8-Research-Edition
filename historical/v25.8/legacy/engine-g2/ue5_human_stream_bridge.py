#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UE5 human stream bridge.

Reads latest MVS frames from /dev/shm/mvs_latest_*.mmap and human_result_*.jsonl.

Video output supports two modes:
  1) legacy ffmpeg MPEG-TS/H.264 mosaic output;
  2) raw per-camera MJPEG over HTTP/TCP, for UE5 DynamicTexture receivers.

The human WebSocket path is kept unchanged: it still broadcasts transformed
human mask polygons / bbox / P labels through ws://host:port when enabled.

This process does not capture cameras and does not run YOLO. It is intentionally
separate from the MVS / human detection process so video encoding or UE5 network
traffic cannot block camera sync, hit judge, or bullet trajectory logic.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from latest_frame_shm import SharedFrameReader
except Exception:
    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from latest_frame_shm import SharedFrameReader


def now_us() -> int:
    return int(time.time() * 1_000_000)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return data


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_config(path: str) -> Dict[str, Any]:
    cfg = load_json(path)
    active = str(cfg.get("active_profile", "") or "").strip()
    if active:
        profiles = cfg.get("profiles", {}) or {}
        if isinstance(profiles, dict) and active in profiles:
            merged = copy.deepcopy(cfg)
            deep_update(merged, profiles[active] or {})
            merged["active_profile"] = active
            return merged
    return cfg


def as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return bool(default)


def expand_path(template: str, root: str, camera: str) -> str:
    p = str(template or "").format(root=root, camera=camera, cam=camera, id=camera)
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    return p


def as_positive_int(v: Any, default: int = 0) -> int:
    try:
        iv = int(v or 0)
    except Exception:
        return int(default)
    return iv if iv > 0 else int(default)


def choose_auto_mosaic_layout(num_cameras: int) -> Tuple[int, int]:
    """Return a safe default mosaic layout for the requested camera count.

    The old default in the config is often 3x2, which only fits 6 streams.
    This helper keeps the familiar compact layouts for small counts and maps
    A-H / 8 cameras to 4x2 so the third row broadcast crash cannot occur.
    """
    n = max(1, int(num_cameras or 0))
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 2, 1
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    if n <= 8:
        return 4, 2
    if n <= 12:
        return 4, 3
    if n <= 16:
        return 4, 4

    # Fallback for future expansion beyond 16 streams.  Use a near-square
    # layout, with rows expanded as needed to guarantee enough slots.
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(float(n) / float(cols))))
    return cols, rows


def resolve_mosaic_layout(
    num_cameras: int,
    arg_cols: int,
    arg_rows: int,
    cfg_cols: Any,
    cfg_rows: Any,
) -> Tuple[int, int, str]:
    """Resolve mosaic rows/cols without allowing capacity < camera count.

    Precedence:
      1. explicit CLI --layout-cols/--layout-rows when provided;
      2. config layout only if it is complete and large enough;
      3. automatic safe layout.

    If a user gives only cols or only rows, the missing dimension is computed.
    If both CLI dimensions are too small, rows are expanded instead of crashing.
    """
    n = max(1, int(num_cameras or 0))
    arg_cols = as_positive_int(arg_cols)
    arg_rows = as_positive_int(arg_rows)
    cfg_cols = as_positive_int(cfg_cols)
    cfg_rows = as_positive_int(cfg_rows)
    auto_cols, auto_rows = choose_auto_mosaic_layout(n)

    source = "auto"
    if arg_cols > 0 and arg_rows > 0:
        cols, rows = arg_cols, arg_rows
        source = "cli"
    elif arg_cols > 0:
        cols = arg_cols
        rows = int(math.ceil(float(n) / float(cols)))
        source = "cli-cols"
    elif arg_rows > 0:
        rows = arg_rows
        cols = int(math.ceil(float(n) / float(rows)))
        source = "cli-rows"
    elif cfg_cols > 0 and cfg_rows > 0 and cfg_cols * cfg_rows >= n:
        cols, rows = cfg_cols, cfg_rows
        source = "config"
    elif cfg_cols > 0 and cfg_rows <= 0:
        cols = cfg_cols
        rows = int(math.ceil(float(n) / float(cols)))
        source = "config-cols"
    elif cfg_rows > 0 and cfg_cols <= 0:
        rows = cfg_rows
        cols = int(math.ceil(float(n) / float(rows)))
        source = "config-rows"
    else:
        cols, rows = auto_cols, auto_rows
        if cfg_cols > 0 and cfg_rows > 0 and cfg_cols * cfg_rows < n:
            source = "auto-config-too-small"

    if cols * rows < n:
        old_cols, old_rows = cols, rows
        # Never crash from an undersized layout.  If columns were explicitly
        # chosen, keep them and add rows; otherwise use the safe auto layout.
        if arg_cols > 0 or (cfg_cols > 0 and source in {"config-cols"}):
            rows = int(math.ceil(float(n) / float(cols)))
        elif arg_rows > 0 or (cfg_rows > 0 and source in {"config-rows"}):
            cols = int(math.ceil(float(n) / float(rows)))
        else:
            cols, rows = auto_cols, auto_rows
            if cols * rows < n:
                rows = int(math.ceil(float(n) / float(cols)))
        source = f"{source}+expanded({old_cols}x{old_rows})"

    return max(1, int(cols)), max(1, int(rows)), source


class HumanJsonlTail:
    def __init__(self, path: str):
        self.path = path
        self.fp = None
        self.pos = 0
        self.inode = None
        self.latest: Optional[Dict[str, Any]] = None

    def _open_if_needed(self) -> bool:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return False
        inode = (st.st_dev, st.st_ino)
        if self.fp is None or self.inode != inode or st.st_size < self.pos:
            self.close()
            self.fp = open(self.path, "r", encoding="utf-8")
            self.inode = inode
            self.pos = 0
        return True

    def read_newest(self) -> Optional[Dict[str, Any]]:
        if not self._open_if_needed() or self.fp is None:
            return self.latest
        self.fp.seek(self.pos)
        newest = None
        while True:
            line = self.fp.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    newest = obj
            except Exception:
                continue
        self.pos = self.fp.tell()
        if newest is not None:
            self.latest = newest
        return self.latest

    def close(self) -> None:
        if self.fp is not None:
            try:
                self.fp.close()
            except Exception:
                pass
        self.fp = None


class WsBroadcaster:
    def __init__(self, host: str, port: int, max_queue: int = 128):
        self.host = host
        self.port = int(port)
        self.max_queue = int(max_queue)
        self.clients = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue] = None
        self.thread: Optional[threading.Thread] = None
        self.started = threading.Event()
        self.stopped = threading.Event()

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)

    async def _broadcast_loop(self):
        assert self.queue is not None
        while True:
            msg = await self.queue.get()
            if msg is None:
                break
            if not self.clients:
                continue
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    async def _run(self):
        import websockets
        self.queue = asyncio.Queue(maxsize=self.max_queue)
        async with websockets.serve(self._handler, self.host, self.port, max_size=8_000_000):
            print(f"[ue5_ws] serving ws://{self.host}:{self.port}", flush=True)
            self.started.set()
            await self._broadcast_loop()

    def start(self) -> None:
        def _target():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self._run())
            finally:
                self.stopped.set()
        self.thread = threading.Thread(target=_target, name="ue5_ws", daemon=True)
        self.thread.start()
        self.started.wait(timeout=5.0)

    def send(self, obj: Dict[str, Any]) -> None:
        if self.loop is None or self.queue is None:
            return
        msg = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        def _put():
            if self.queue is None:
                return
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
            try:
                self.queue.put_nowait(msg)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(_put)

    def stop(self) -> None:
        if self.loop is not None and self.queue is not None:
            def _request_stop():
                if self.queue is None:
                    return
                try:
                    while self.queue.full():
                        self.queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.queue.put_nowait(None)
                except Exception:
                    pass
                for ws in list(self.clients):
                    try:
                        self.loop.create_task(ws.close())
                    except Exception:
                        pass
            try:
                self.loop.call_soon_threadsafe(_request_stop)
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)




def ensure_bgr_uint8(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return a contiguous uint8 BGR frame suitable for cv2.imencode."""
    if frame is None:
        return None
    if not isinstance(frame, np.ndarray):
        return None
    if frame.dtype != np.uint8:
        try:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        except Exception:
            return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] < 3:
        return None
    else:
        frame = frame[:, :, :3]
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


class MjpegFrameHub:
    """Thread-safe latest-JPEG store used by the HTTP MJPEG server.

    The bridge updates one JPEG per camera.  HTTP clients only receive the
    latest frame, so a slow UE/browser client cannot block MVS capture, human
    JSONL processing, hit judge, or any other existing pipeline.
    """

    def __init__(self, cameras: List[str], jpeg_quality: int = 75, resize_w: int = 0, resize_h: int = 0):
        self.cameras = [str(c) for c in cameras]
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.resize_w = int(resize_w or 0)
        self.resize_h = int(resize_h or 0)
        self._cond = threading.Condition()
        self._stopping = False
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._seq: Dict[str, int] = {c: 0 for c in self.cameras}

    def normalize_camera(self, camera: str) -> Optional[str]:
        camera = str(camera or "").strip()
        if camera in self.cameras:
            return camera
        if len(self.cameras) == 1 and camera in {"", "default", "latest", "stream"}:
            return self.cameras[0]
        return None

    def update(self, camera: str, frame: np.ndarray, frame_id: int = -1, timestamp_us: int = 0) -> None:
        camera = self.normalize_camera(camera)
        if camera is None:
            return
        bgr = ensure_bgr_uint8(frame)
        if bgr is None:
            return
        if self.resize_w > 0 and self.resize_h > 0:
            bgr = cv2.resize(bgr, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        jpeg = enc.tobytes()
        with self._cond:
            self._seq[camera] = int(self._seq.get(camera, 0)) + 1
            self._latest[camera] = {
                "jpeg": jpeg,
                "seq": self._seq[camera],
                "frame_id": int(frame_id),
                "timestamp_us": int(timestamp_us or 0),
                "width": int(w),
                "height": int(h),
                "camera": camera,
                "updated_wall_us": now_us(),
            }
            self._cond.notify_all()

    def wait_frame(self, camera: str, last_seq: int, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        camera = self.normalize_camera(camera)
        if camera is None:
            return None
        deadline = time.time() + float(timeout)
        with self._cond:
            while not self._stopping:
                item = self._latest.get(camera)
                if item is not None and int(item.get("seq", 0)) != int(last_seq):
                    return dict(item)
                remain = deadline - time.time()
                if remain <= 0:
                    return dict(item) if item is not None else None
                self._cond.wait(timeout=min(0.25, remain))
        return None

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()


class MjpegHttpServer:
    def __init__(self, host: str, port: int, path_prefix: str, hub: MjpegFrameHub):
        self.host = str(host or "0.0.0.0")
        self.port = int(port)
        self.path_prefix = "/" + str(path_prefix or "/mjpeg").strip("/")
        self.hub = hub
        self._stop_event = threading.Event()
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        hub = self.hub
        prefix = self.path_prefix
        cameras = list(hub.cameras)
        stop_event = self._stop_event

        class ReusableThreadingHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True
            block_on_close = False

        class Handler(BaseHTTPRequestHandler):
            server_version = "YSXQRawMjpeg/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                # Avoid per-frame HTTP spam.  Connection open/close is logged by the bridge.
                return

            def _send_text(self, code: int, text: str) -> None:
                data = text.encode("utf-8", errors="replace")
                self.send_response(code)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _camera_from_request(self) -> Optional[str]:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")
                qs = parse_qs(parsed.query or "")
                if "camera" in qs and qs["camera"]:
                    return hub.normalize_camera(qs["camera"][0])
                if path == prefix:
                    return hub.normalize_camera("")
                base = prefix + "/"
                if path.startswith(base):
                    cam = path[len(base):].split("/", 1)[0]
                    if cam.endswith(".mjpg") or cam.endswith(".mjpeg"):
                        cam = cam.rsplit(".", 1)[0]
                    return hub.normalize_camera(cam)
                return None

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path.rstrip("/") in {"", "/", "/health"}:
                    self._send_text(200, "OK\npaths:\n" + "\n".join(f"{prefix}/{c}" for c in cameras) + "\n")
                    return
                camera = self._camera_from_request()
                if camera is None:
                    self._send_text(404, "Unknown MJPEG path. Use one of:\n" + "\n".join(f"{prefix}/{c}" for c in cameras) + "\n")
                    return
                print(f"[mjpeg_http] client {self.client_address[0]} connected camera={camera}", flush=True)
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_seq = -1
                try:
                    while not stop_event.is_set():
                        item = hub.wait_frame(camera, last_seq, timeout=0.5)
                        if item is None:
                            continue
                        last_seq = int(item.get("seq", last_seq))
                        jpeg = item.get("jpeg", b"")
                        if not jpeg:
                            continue
                        headers = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n".encode("ascii")
                            + f"X-Pair-Id: {camera}\r\n".encode("ascii")
                            + f"X-Stream-Frame-Index: {last_seq}\r\n".encode("ascii")
                            + f"X-Mvs-Frame-Num: {int(item.get('frame_id', -1))}\r\n".encode("ascii")
                            + f"X-Timestamp-Us: {int(item.get('timestamp_us', 0))}\r\n".encode("ascii")
                            + f"X-Frame-Width: {int(item.get('width', 0))}\r\n".encode("ascii")
                            + f"X-Frame-Height: {int(item.get('height', 0))}\r\n".encode("ascii")
                            + b"\r\n"
                        )
                        self.wfile.write(headers)
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    print(f"[mjpeg_http][WARN] client error camera={camera}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    print(f"[mjpeg_http] client {self.client_address[0]} disconnected camera={camera}", flush=True)

        self.httpd = ReusableThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="mjpeg_http", daemon=True)
        self.thread.start()
        paths = ", ".join(f"http://{self.host}:{self.port}{self.path_prefix}/{c}" for c in cameras)
        print(f"[mjpeg_http] serving {paths} quality={hub.jpeg_quality}", flush=True)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.hub.stop()
        except Exception:
            pass
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
            try:
                self.httpd.server_close()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class ShmFrameSlot:
    def __init__(self, camera: str, name_template: str):
        self.camera = camera
        self.name = name_template.format(camera=camera, cam=camera, id=camera)
        self.reader: Optional[SharedFrameReader] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_id: int = -1
        self.timestamp_us: int = 0
        self.last_open_try = 0.0

    def read(self) -> Optional[np.ndarray]:
        if self.reader is None:
            now = time.time()
            if now - self.last_open_try < 0.5:
                return self.frame
            self.last_open_try = now
            try:
                self.reader = SharedFrameReader(self.name)
                print(f"[ue5_bridge][{self.camera}] opened /dev/shm/{self.name}.mmap", flush=True)
            except Exception:
                return self.frame
        try:
            item = self.reader.read_latest()
        except Exception as exc:
            print(f"[ue5_bridge][{self.camera}][WARN] shm read failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
            return self.frame
        if item is not None:
            self.frame = item.get("frame")
            self.frame_id = int(item.get("frame_id", self.frame_id))
            self.timestamp_us = int(item.get("timestamp_us", 0) or 0)
        return self.frame

    def close(self) -> None:
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None


def fit_into_tile(frame: np.ndarray, tile_w: int, tile_h: int) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = frame.shape[:2]
    scale = min(float(tile_w) / max(1, w), float(tile_h) / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    off_x = (tile_w - new_w) // 2
    off_y = (tile_h - new_h) // 2
    tile[off_y:off_y + new_h, off_x:off_x + new_w] = resized[:, :, :3]
    return tile, {"scale": scale, "off_x": float(off_x), "off_y": float(off_y), "src_w": float(w), "src_h": float(h)}


def transform_xy(x: float, y: float, tile_origin: Tuple[int, int], fit: Dict[str, float]) -> List[int]:
    return [
        int(round(tile_origin[0] + fit["off_x"] + x * fit["scale"])),
        int(round(tile_origin[1] + fit["off_y"] + y * fit["scale"])),
    ]


def transform_bbox(bbox: List[Any], tile_origin: Tuple[int, int], fit: Dict[str, float]) -> List[int]:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return [0, 0, 0, 0]
    p1 = transform_xy(float(bbox[0]), float(bbox[1]), tile_origin, fit)
    p2 = transform_xy(float(bbox[2]), float(bbox[3]), tile_origin, fit)
    return [p1[0], p1[1], p2[0], p2[1]]


def transform_polygons(polys: Any, tile_origin: Tuple[int, int], fit: Dict[str, float], max_points: int = 240) -> List[List[List[int]]]:
    out: List[List[List[int]]] = []
    if not isinstance(polys, list):
        return out
    for poly in polys:
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        step = max(1, int(np.ceil(len(poly) / max(3, max_points))))
        pts: List[List[int]] = []
        for pt in poly[::step]:
            if isinstance(pt, list) and len(pt) >= 2:
                pts.append(transform_xy(float(pt[0]), float(pt[1]), tile_origin, fit))
        if len(pts) >= 3:
            out.append(pts)
    return out


def label_for_person(p: Dict[str, Any]) -> str:
    gid = p.get("global_id_from_A", None)
    if gid is not None:
        try:
            return f"P{int(gid)}"
        except Exception:
            pass
    lid = p.get("local_stable_id", p.get("stable_id", p.get("person_id", None)))
    if lid is not None:
        try:
            return f"L{int(lid)}"
        except Exception:
            pass
    return "UNK"


def start_ffmpeg(width: int, height: int, fps: float, url: str, codec: str, preset: str, bitrate: str, extra_args: List[str]) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
        "-an", "-c:v", codec,
    ]
    if codec == "libx264":
        cmd += ["-preset", preset, "-tune", "zerolatency"]
    if bitrate:
        cmd += ["-b:v", bitrate]
    cmd += ["-pix_fmt", "yuv420p"]
    cmd += extra_args
    cmd += ["-f", "mpegts", url]
    print("[ffmpeg] " + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bridge MVS shm + human JSONL to UE5 video + WebSocket overlay")
    ap.add_argument("--config", default="ids_8cam_fusion_config.json")
    ap.add_argument("--cameras", default="")
    ap.add_argument("--sync-root", default="")
    ap.add_argument("--tile-width", type=int, default=0)
    ap.add_argument("--tile-height", type=int, default=0)
    ap.add_argument("--layout-cols", type=int, default=0)
    ap.add_argument("--layout-rows", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--ffmpeg-url", default="")
    ap.add_argument("--ws-host", default="")
    ap.add_argument("--ws-port", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--mjpeg-host", default="")
    ap.add_argument("--mjpeg-port", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}
    # Supports both the unified 6-12 config and the generated MVS-only runtime config.
    mvs_cfg = cfg.get("mvs_config", cfg) if isinstance(cfg.get("mvs_config", cfg), dict) else cfg
    sync_cfg = mvs_cfg.get("sync", {}) if isinstance(mvs_cfg.get("sync", {}), dict) else {}
    bridge_cfg = cfg.get("ue5_human_stream", {}) if isinstance(cfg.get("ue5_human_stream", {}), dict) else {}

    cameras = [c.strip() for c in (args.cameras or ",".join(bridge_cfg.get("cameras", ["A", "B", "C", "D", "E"]))).split(",") if c.strip()]
    sync_root = os.path.abspath(args.sync_root or bridge_cfg.get("sync_root") or sync_cfg.get("root") or "./sync_ipc")
    name_template = str(sync_cfg.get("mvs_frame_pub_name_template", "mvs_latest_{camera}") or "mvs_latest_{camera}")
    human_template = str(sync_cfg.get("human_jsonl_template", "{root}/human_result_{camera}.jsonl") or "{root}/human_result_{camera}.jsonl")

    tile_w = int(args.tile_width or bridge_cfg.get("tile_width", 640) or 640)
    tile_h = int(args.tile_height or bridge_cfg.get("tile_height", 488) or 488)
    cols, rows, layout_source = resolve_mosaic_layout(
        len(cameras),
        args.layout_cols,
        args.layout_rows,
        bridge_cfg.get("layout_cols", 0),
        bridge_cfg.get("layout_rows", 0),
    )
    fps = float(args.fps or bridge_cfg.get("fps", 25.0) or 25.0)
    video_enable = (not args.no_video) and as_bool(bridge_cfg.get("video_enable", True), True)
    ws_enable = (not args.no_ws) and as_bool(bridge_cfg.get("ws_enable", True), True)
    video_transport = str(bridge_cfg.get("video_transport", bridge_cfg.get("transport", "ffmpeg_mpegts")) or "ffmpeg_mpegts").strip().lower()
    mjpeg_enable = video_enable and video_transport in {"mjpeg", "mjpeg_http", "mjpeg_http_per_camera", "per_camera_mjpeg", "raw_mjpeg"}
    ffmpeg_video_enable = video_enable and not mjpeg_enable
    ffmpeg_url = str(args.ffmpeg_url or bridge_cfg.get("ffmpeg_url", "udp://127.0.0.1:5000?pkt_size=1316") or "")
    codec = str(bridge_cfg.get("ffmpeg_codec", "libx264") or "libx264")
    preset = str(bridge_cfg.get("ffmpeg_preset", "ultrafast") or "ultrafast")
    bitrate = str(bridge_cfg.get("ffmpeg_bitrate", "8000k") or "8000k")
    ws_host = str(args.ws_host or bridge_cfg.get("ws_host", "0.0.0.0") or "0.0.0.0")
    ws_port = int(args.ws_port or bridge_cfg.get("ws_port", 8766) or 8766)
    hold_ms = float(bridge_cfg.get("human_hold_ms", 250) or 250)
    send_empty_frames = as_bool(bridge_cfg.get("send_empty_frames", True), True)
    draw_camera_labels = as_bool(bridge_cfg.get("draw_camera_labels", True), True)
    max_polygon_points = int(bridge_cfg.get("max_polygon_points", 240) or 240)
    mjpeg_host = str(args.mjpeg_host or bridge_cfg.get("mjpeg_host", bridge_cfg.get("http_host", "0.0.0.0")) or "0.0.0.0")
    mjpeg_port = int(args.mjpeg_port or bridge_cfg.get("mjpeg_port", bridge_cfg.get("http_port", 18080)) or 18080)
    mjpeg_path_prefix = str(bridge_cfg.get("mjpeg_path_prefix", bridge_cfg.get("mjpeg_path", "/mjpeg")) or "/mjpeg")
    mjpeg_quality = int(bridge_cfg.get("mjpeg_quality", bridge_cfg.get("jpeg_quality", 75)) or 75)
    mjpeg_resize_w = int(bridge_cfg.get("mjpeg_resize_width", bridge_cfg.get("raw_stream_width", 0)) or 0)
    mjpeg_resize_h = int(bridge_cfg.get("mjpeg_resize_height", bridge_cfg.get("raw_stream_height", 0)) or 0)

    mosaic_w = cols * tile_w
    mosaic_h = rows * tile_h
    print(f"[ue5_bridge] cameras={cameras} sync_root={sync_root}", flush=True)
    print(f"[ue5_bridge] mosaic={mosaic_w}x{mosaic_h} tile={tile_w}x{tile_h} layout={cols}x{rows} source={layout_source} fps={fps:.2f}", flush=True)
    print(f"[ue5_bridge] video_transport={video_transport} raw_mjpeg={mjpeg_enable} ffmpeg_video={ffmpeg_video_enable}", flush=True)

    frame_slots = {c: ShmFrameSlot(c, name_template) for c in cameras}
    human_tails = {c: HumanJsonlTail(expand_path(human_template, sync_root, c)) for c in cameras}
    latest_human: Dict[str, Optional[Dict[str, Any]]] = {c: None for c in cameras}
    latest_human_wall_us: Dict[str, int] = {c: 0 for c in cameras}

    ws = None
    if ws_enable:
        ws = WsBroadcaster(ws_host, ws_port)
        ws.start()
        ws.send({
            "type": "overlay_human_layout",
            "source_width": mosaic_w,
            "source_height": mosaic_h,
            "tile_width": tile_w,
            "tile_height": tile_h,
            "layout_cols": cols,
            "layout_rows": rows,
            "cameras": cameras,
        })

    mjpeg_hub = None
    mjpeg_server = None
    if mjpeg_enable:
        mjpeg_hub = MjpegFrameHub(cameras, jpeg_quality=mjpeg_quality, resize_w=mjpeg_resize_w, resize_h=mjpeg_resize_h)
        mjpeg_server = MjpegHttpServer(mjpeg_host, mjpeg_port, mjpeg_path_prefix, mjpeg_hub)
        mjpeg_server.start()
        if cameras:
            print(f"[ue5_bridge] MJPEG URL example: http://192.168.31.115:{mjpeg_port}{mjpeg_path_prefix}/{cameras[0]}", flush=True)

    ffmpeg = None
    if ffmpeg_video_enable:
        if not ffmpeg_url:
            print("[ue5_bridge][WARN] ffmpeg video enabled but ffmpeg_url is empty; disable ffmpeg video", flush=True)
            ffmpeg_video_enable = False
        else:
            ffmpeg = start_ffmpeg(mosaic_w, mosaic_h, fps, ffmpeg_url, codec, preset, bitrate, list(bridge_cfg.get("ffmpeg_extra_args", []) or []))

    frame_interval = 1.0 / max(1.0, fps)
    next_tick = time.perf_counter()
    frames_sent = 0
    last_print = time.time()

    try:
        while True:
            tick_start = time.perf_counter()
            mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
            layout: Dict[str, Dict[str, Any]] = {}

            for idx, cam in enumerate(cameras):
                row = idx // cols
                col = idx % cols
                x0 = col * tile_w
                y0 = row * tile_h
                if y0 + tile_h > mosaic_h or x0 + tile_w > mosaic_w:
                    print(
                        f"[ue5_bridge][WARN] skip camera={cam}: tile origin=({x0},{y0}) "
                        f"outside mosaic={mosaic_w}x{mosaic_h}; layout={cols}x{rows}",
                        flush=True,
                    )
                    continue
                frame = frame_slots[cam].read()
                if frame is not None:
                    if mjpeg_hub is not None:
                        mjpeg_hub.update(cam, frame, frame_id=frame_slots[cam].frame_id, timestamp_us=frame_slots[cam].timestamp_us)
                    tile, fit = fit_into_tile(frame, tile_w, tile_h)
                    mosaic[y0:y0 + tile_h, x0:x0 + tile_w] = tile
                else:
                    fit = {"scale": 1.0, "off_x": 0.0, "off_y": 0.0, "src_w": float(tile_w), "src_h": float(tile_h)}
                    cv2.putText(mosaic, f"WAIT {cam}", (x0 + 24, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
                if draw_camera_labels:
                    cv2.putText(mosaic, cam, (x0 + 12, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                layout[cam] = {
                    "x": x0,
                    "y": y0,
                    "w": tile_w,
                    "h": tile_h,
                    "fit": fit,
                    "frame_id": frame_slots[cam].frame_id,
                    "timestamp_us": frame_slots[cam].timestamp_us,
                }

            now_wall_us = now_us()
            persons_out: List[Dict[str, Any]] = []
            for cam in cameras:
                rec = human_tails[cam].read_newest()
                if rec is not None:
                    latest_human[cam] = rec
                    try:
                        latest_human_wall_us[cam] = int(rec.get("record_wall_us", now_wall_us) or now_wall_us)
                    except Exception:
                        latest_human_wall_us[cam] = now_wall_us
                rec = latest_human.get(cam)
                if not isinstance(rec, dict):
                    continue
                age_ms = (now_wall_us - int(latest_human_wall_us.get(cam, 0) or 0)) / 1000.0
                if age_ms > hold_ms:
                    continue
                info = layout.get(cam)
                if not info:
                    continue
                fit = info["fit"]
                origin = (int(info["x"]), int(info["y"]))
                frame_w = int(rec.get("frame_w", fit.get("src_w", tile_w)) or tile_w)
                frame_h = int(rec.get("frame_h", fit.get("src_h", tile_h)) or tile_h)
                for p in rec.get("persons", []) or []:
                    if not isinstance(p, dict):
                        continue
                    outp = {
                        "camera": cam,
                        "sync_index": rec.get("sync_index_hint"),
                        "mvs_frame_num": rec.get("mvs_frame_num"),
                        "frame_id_local": rec.get("frame_id_local"),
                        "global_id_from_A": p.get("global_id_from_A"),
                        "local_stable_id": p.get("local_stable_id", p.get("stable_id")),
                        "label": label_for_person(p),
                        "confidence": p.get("confidence"),
                        "local_xy": p.get("local_xy"),
                        "source_camera_frame_w": frame_w,
                        "source_camera_frame_h": frame_h,
                        "original_bbox_xyxy": p.get("bbox_xyxy"),
                        "bbox_xyxy": transform_bbox(p.get("bbox_xyxy", []), origin, fit),
                        "mask_polygons": transform_polygons(p.get("mask_polygons"), origin, fit, max_points=max_polygon_points),
                    }
                    persons_out.append(outp)

            if ws is not None and (persons_out or send_empty_frames):
                ws.send({
                    "type": "overlay_human_frame",
                    "source_width": mosaic_w,
                    "source_height": mosaic_h,
                    "wall_us": now_wall_us,
                    "persons": persons_out,
                    "layout": {cam: {k: v for k, v in info.items() if k != "fit"} for cam, info in layout.items()},
                })

            if ffmpeg_video_enable and ffmpeg is not None and ffmpeg.stdin is not None:
                try:
                    ffmpeg.stdin.write(mosaic.tobytes())
                    frames_sent += 1
                except BrokenPipeError:
                    print("[ffmpeg][WARN] pipe broken, restarting in 1s", flush=True)
                    try:
                        ffmpeg.kill()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    ffmpeg = start_ffmpeg(mosaic_w, mosaic_h, fps, ffmpeg_url, codec, preset, bitrate, list(bridge_cfg.get("ffmpeg_extra_args", []) or []))

            if time.time() - last_print >= 1.0:
                dt = max(1e-6, time.time() - last_print)
                print(f"[ue5_bridge][perf] video_fps={frames_sent/dt:.1f} persons={len(persons_out)} ws_clients={len(ws.clients) if ws else 0}", flush=True)
                frames_sent = 0
                last_print = time.time()

            next_tick += frame_interval
            remain = next_tick - time.perf_counter()
            if remain > 0:
                time.sleep(remain)
            elif remain < -frame_interval * 2:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("[ue5_bridge] interrupted", flush=True)
    finally:
        for tail in human_tails.values():
            tail.close()
        for slot in frame_slots.values():
            slot.close()
        if mjpeg_server is not None:
            mjpeg_server.stop()
        if ffmpeg is not None:
            try:
                if ffmpeg.stdin:
                    ffmpeg.stdin.close()
            except Exception:
                pass
            try:
                ffmpeg.terminate()
                ffmpeg.wait(timeout=2.0)
            except Exception:
                try:
                    ffmpeg.kill()
                except Exception:
                    pass
        if ws is not None:
            ws.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
