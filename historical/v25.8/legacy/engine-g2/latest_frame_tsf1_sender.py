"""
latest_frame_tsf1_sender.py  —  从共享内存读取最新帧，按 TSF1 协议发送

改动说明（相对原版）：
  1. TSF1 协议 header 新增 4 字节 CRC32 校验字段（payload 的 CRC32），
     接收端可用来检测传输损坏，字段位于 header 末尾。
     新 header 格式：magic(4) version(1) w(2) h(2) ts(8) fid(8) codec(1)
                     payload_len(4) cam_len(2) crc32(4)  = 36 bytes
  2. LatestPacketSender 新增 ConnState 枚举，外部可查询连接状态。
  3. 连接/断线日志统一，避免日志风暴（1 秒节流保持不变）。
"""

from __future__ import annotations

import argparse
import enum
import socket
import struct
import threading
import time
import zlib
from typing import Optional

import cv2
import numpy as np

from latest_frame_shm import SharedFrameReader

# ---------------------------------------------------------------------------
# TSF1 协议常量
# ---------------------------------------------------------------------------
TS_MAGIC          = b"TSF1"
TS_VERSION        = 1
TS_CODEC_RAW      = 0
TS_CODEC_ZLIB     = 1

# 新增 crc32(4) 字段，放在 header 末尾
# magic(4) version(1) w(2) h(2) timestamp_us(8) frame_id(8)
# codec(1) payload_len(4) camera_len(2) crc32(4)
TS_HEADER_STRUCT  = struct.Struct("!4sBHHQQBIHI")
TS_HEADER_SIZE    = TS_HEADER_STRUCT.size   # 36 bytes


# ---------------------------------------------------------------------------
# 连接状态
# ---------------------------------------------------------------------------
class ConnState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING   = "connecting"
    CONNECTED    = "connected"


# ---------------------------------------------------------------------------
# 异步发送器：只保留最新包，丢弃积压的旧包
# ---------------------------------------------------------------------------
class LatestPacketSender:
    def __init__(self, server_ip: str, server_port: int):
        self.server_ip   = server_ip
        self.server_port = int(server_port)

        self._lock   = threading.Lock()
        self._cond   = threading.Condition(self._lock)
        self._pending: Optional[bytes] = None
        self._stop   = False

        self._conn_state: ConnState = ConnState.DISCONNECTED
        self._sock: Optional[socket.socket] = None
        self._last_err_ts = 0.0

        self._thread = threading.Thread(target=self._worker, daemon=True)

    # --- 公共 API -----------------------------------------------------------

    @property
    def conn_state(self) -> ConnState:
        """线程安全地读取当前连接状态。"""
        with self._lock:
            return self._conn_state

    def start(self) -> None:
        self._thread.start()

    def submit(self, packet: bytes) -> None:
        """提交最新包；若已有待发包则直接替换（丢弃旧包）。"""
        with self._cond:
            self._pending = packet
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._close_sock()

    # --- 内部方法 -----------------------------------------------------------

    def _set_state(self, state: ConnState) -> None:
        self._conn_state = state   # 在 _lock 持有期间调用

    def _close_sock(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def _connect(self) -> None:
        with self._lock:
            self._set_state(ConnState.CONNECTING)
        while not self._stop:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self.server_ip, self.server_port))
                with self._lock:
                    self._sock = s
                    self._set_state(ConnState.CONNECTED)
                print(f"[tsf1_sender] connected → {self.server_ip}:{self.server_port}")
                return
            except Exception as e:
                now = time.time()
                if now - self._last_err_ts > 1.0:
                    self._last_err_ts = now
                time.sleep(0.5)

    def _worker(self) -> None:
        self._connect()
        while not self._stop:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.2)
                if self._stop:
                    break
                packet        = self._pending
                self._pending = None

            if packet is None:
                continue

            # 若 sock 为 None，先重连
            with self._lock:
                sock = self._sock
            if sock is None:
                self._connect()
                with self._lock:
                    sock = self._sock
                if sock is None:
                    continue

            try:
                sock.sendall(struct.pack("!I", len(packet)))
                sock.sendall(packet)
            except Exception as e:
                now = time.time()
                if now - self._last_err_ts > 1.0:
                    print(f"[tsf1_sender] send failed: {e}")
                    self._last_err_ts = now
                with self._lock:
                    self._close_sock()
                    self._set_state(ConnState.DISCONNECTED)
                self._connect()

        with self._lock:
            self._set_state(ConnState.DISCONNECTED)


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def build_tsf1_packet(
    ts_img:       np.ndarray,
    timestamp_us: int,
    frame_id:     int,
    camera_id:    str,
    codec:        str = "zlib",
    zlib_level:   int = 1,
) -> bytes:
    """
    构造带 CRC32 校验的 TSF1 数据包。
    CRC32 覆盖范围：payload（压缩或原始像素）。
    """
    # 确保单通道 uint8
    if ts_img.ndim == 3:
        ts_img = ts_img[:, :, 0] if ts_img.shape[2] == 1 else cv2.cvtColor(ts_img, cv2.COLOR_BGR2GRAY)
    if ts_img.dtype != np.uint8:
        ts_img = ts_img.astype(np.uint8)

    h, w = ts_img.shape[:2]
    raw  = ts_img.tobytes()
    camera_bytes = camera_id.encode("utf-8", errors="replace")

    if codec == "raw":
        codec_id = TS_CODEC_RAW
        payload  = raw
    else:
        codec_id = TS_CODEC_ZLIB
        payload  = zlib.compress(raw, level=max(0, min(9, int(zlib_level))))

    crc32_val = zlib.crc32(payload) & 0xFFFFFFFF

    header = TS_HEADER_STRUCT.pack(
        TS_MAGIC,
        TS_VERSION,
        int(w),
        int(h),
        int(timestamp_us),
        int(frame_id),
        int(codec_id),
        int(len(payload)),
        int(len(camera_bytes)),
        crc32_val,
    )
    return header + camera_bytes + payload


# ---------------------------------------------------------------------------
# 启动时等待共享内存就绪
# ---------------------------------------------------------------------------
def open_reader_with_retry(name: str, retry_sec: float = 0.5) -> SharedFrameReader:
    last_print_ts = 0.0
    while True:
        try:
            reader = SharedFrameReader(name)
            print(f"[tsf1_sender] shared memory ready: {name}")
            return reader
        except FileNotFoundError:
            now = time.time()
            if now - last_print_ts > 1.0:
                print(f"[tsf1_sender] waiting for shared memory: {name}")
                last_print_ts = now
            time.sleep(retry_sec)
        except Exception as e:
            now = time.time()
            if now - last_print_ts > 1.0:
                print(f"[tsf1_sender] open shared memory failed: {e}")
                last_print_ts = now
            time.sleep(retry_sec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="从共享内存读取最新单通道帧，按 TSF1 协议（含 CRC32）发送"
    )
    ap.add_argument("--frame-pub-name", type=str, default="bullet_ts_latest",
                    help="共享内存名字")
    ap.add_argument("--server-ip",   type=str, required=True,  help="服务器 IP")
    ap.add_argument("--server-port", type=int, default=5001,   help="服务器端口")
    ap.add_argument("--camera-id",   type=str, default="cam0", help="camera_id 标识")
    ap.add_argument("--send-fps",    type=float, default=8.0,  help="发送帧率上限")
    ap.add_argument("--codec",       type=str, default="zlib",
                    choices=["raw", "zlib"],                   help="TSF1 payload 编码")
    ap.add_argument("--zlib-level",  type=int, default=1,      help="zlib 压缩等级 0-9")
    ap.add_argument("--show-local",  action="store_true",      help="本地预览发送出去的单通道图")
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    reader = open_reader_with_retry(args.frame_pub_name)
    sender = LatestPacketSender(args.server_ip, args.server_port)
    sender.start()

    interval      = 1.0 / max(0.1, float(args.send_fps))
    last_send_wall = 0.0
    sent_cnt       = 0

    try:
        while True:
            now = time.time()
            if now - last_send_wall < interval:
                time.sleep(0.001)
                continue

            item = reader.read_latest()
            if item is None:
                time.sleep(0.002)
                continue

            frame = item["frame"]
            if frame.ndim == 3 and frame.shape[2] == 1:
                ts_img = frame[:, :, 0]
            elif frame.ndim == 2:
                ts_img = frame
            else:
                ts_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            packet = build_tsf1_packet(
                ts_img,
                timestamp_us = int(item["timestamp_us"]),
                frame_id     = int(item["frame_id"]),
                camera_id    = args.camera_id,
                codec        = args.codec,
                zlib_level   = args.zlib_level,
            )
            sender.submit(packet)
            last_send_wall = now
            sent_cnt += 1

            if args.show_local:
                cv2.imshow("tsf1_sender_local", ts_img)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        reader.close()
        sender.close()
        cv2.destroyAllWindows()
        print(f"[tsf1_sender] stopped — sent frames={sent_cnt}")


if __name__ == "__main__":
    main()
