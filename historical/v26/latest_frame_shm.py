"""
latest_frame_shm.py  —  进程间共享内存帧缓冲（双缓冲版）

改动说明（相对原版）：
  - 使用双缓冲（slot 0 / slot 1）彻底消除写端写入期间读端读到撕裂帧的竞争：
      写端通过原子方式切换 active_slot，读端只读 active_slot 指向的那块数据。
  - header 中新增 write_slot（写端正在写哪个 slot）和 active_slot（读端应读哪个 slot），
    以及每个 slot 独立的 frame_id / timestamp。
  - 读端在读取完整帧之前先记录 active_slot，读完后再确认 active_slot 未改变，
    若发生切换则重试，保证 frame 数据与 header 字段一致。

Header 布局（全部 uint32，大小 = HEADER_U32_COUNT * 4 = 128 bytes）：
  [0]  magic
  [1]  version
  [2]  width
  [3]  height
  [4]  channels
  [5]  dtype (固定 1 = uint8)
  [6]  stride
  [7]  write_slot      — 写端当前正在写的 slot (0 或 1)
  [8]  active_slot     — 读端应读的 slot (0 或 1)；写端完成后原子更新
  [9]  data_bytes      — 单个 slot 的数据大小（字节）
  # slot 0 元数据
  [10] slot0_ready     — 1=有效
  [11] slot0_frame_id  (低 32 位)
  [12] slot0_ts_lo
  [13] slot0_ts_hi
  # slot 1 元数据
  [14] slot1_ready
  [15] slot1_frame_id
  [16] slot1_ts_lo
  [17] slot1_ts_hi
  [18..31] 保留
"""

from __future__ import annotations

import mmap
import os
import time
from typing import Optional, Tuple

import numpy as np

MAGIC = 0x46524D32          # 版本标识，与原版 0x46524D31 不同以防混用
VERSION = 2
DTYPE_UINT8 = 1
HEADER_U32_COUNT = 32       # 128 bytes，比原版多一倍，留有充裕余量
HEADER_BYTES = HEADER_U32_COUNT * 4

# header 字段下标
_H_MAGIC        = 0
_H_VERSION      = 1
_H_WIDTH        = 2
_H_HEIGHT       = 3
_H_CHANNELS     = 4
_H_DTYPE        = 5
_H_STRIDE       = 6
_H_WRITE_SLOT   = 7
_H_ACTIVE_SLOT  = 8
_H_DATA_BYTES   = 9
# slot 0
_H_S0_READY     = 10
_H_S0_FRAME_ID  = 11
_H_S0_TS_LO     = 12
_H_S0_TS_HI     = 13
# slot 1
_H_S1_READY     = 14
_H_S1_FRAME_ID  = 15
_H_S1_TS_LO     = 16
_H_S1_TS_HI     = 17

_SLOT_OFFSETS = [
    (_H_S0_READY, _H_S0_FRAME_ID, _H_S0_TS_LO, _H_S0_TS_HI),
    (_H_S1_READY, _H_S1_FRAME_ID, _H_S1_TS_LO, _H_S1_TS_HI),
]


def _split_u64(v: int) -> Tuple[int, int]:
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    return v & 0xFFFFFFFF, (v >> 32) & 0xFFFFFFFF


def _join_u64(lo: int, hi: int) -> int:
    return (int(hi) << 32) | int(lo)


def _backend_path(name: str) -> str:
    return f"/dev/shm/{name}.mmap"


class SharedFrameWriter:
    """
    写端：每次 write() 写入"非活跃"的 slot，写完后原子切换 active_slot。
    读端始终读 active_slot，因此两者之间没有共享的写入窗口。
    """

    def __init__(self, name: str, width: int, height: int, channels: int = 3, create: bool = True):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.stride = self.width * self.channels
        self.data_bytes = self.height * self.stride
        # 总大小 = header + 2 个 slot
        self.total_bytes = HEADER_BYTES + 2 * self.data_bytes
        self.path = _backend_path(name)

        if create:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass

        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(self.fd, self.total_bytes)
        self.mm = mmap.mmap(self.fd, self.total_bytes, access=mmap.ACCESS_WRITE)

        self.header = np.ndarray(
            (HEADER_U32_COUNT,), dtype=np.uint32, buffer=self.mm, offset=0
        )
        # 两个帧缓冲 slot，紧跟 header
        self._slots = [
            np.ndarray(
                (self.height, self.width, self.channels),
                dtype=np.uint8,
                buffer=self.mm,
                offset=HEADER_BYTES + i * self.data_bytes,
            )
            for i in range(2)
        ]

        self._frame_id = 0
        self._write_header_base()

    def _write_header_base(self) -> None:
        self.header[:] = 0
        self.header[_H_MAGIC]      = MAGIC
        self.header[_H_VERSION]    = VERSION
        self.header[_H_WIDTH]      = self.width
        self.header[_H_HEIGHT]     = self.height
        self.header[_H_CHANNELS]   = self.channels
        self.header[_H_DTYPE]      = DTYPE_UINT8
        self.header[_H_STRIDE]     = self.stride
        self.header[_H_WRITE_SLOT] = 0
        self.header[_H_ACTIVE_SLOT]= 0
        self.header[_H_DATA_BYTES] = self.data_bytes
        self.mm.flush()

    def write(self, frame: np.ndarray, timestamp_us: Optional[int] = None, frame_id: Optional[int] = None) -> None:
        if frame.dtype != np.uint8:
            raise ValueError("SharedFrameWriter 只支持 uint8 图像")
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"图像尺寸不匹配: got {frame.shape}, "
                f"expected {(self.height, self.width, self.channels)}"
            )

        # 写入"非活跃" slot，保证读端不会读到写到一半的数据
        active = int(self.header[_H_ACTIVE_SLOT])
        write_slot = 1 - active          # 另一个 slot

        # 标记正在写的 slot
        self.header[_H_WRITE_SLOT] = write_slot

        # 先将目标 slot 的 ready 置 0，防止读端误读旧数据
        h_ready, h_fid, h_ts_lo, h_ts_hi = _SLOT_OFFSETS[write_slot]
        self.header[h_ready] = 0

        # 拷贝像素
        np.copyto(self._slots[write_slot], frame)

        # 更新元数据
        # 默认沿用内部递增帧号；MVS 发布端可以传入相机采集线程的
        # frame_id_local，使 OpenGL 读到的 frame_id 与 human_result.jsonl
        # 的 frame_id_local 完全一致，避免“最新背景 + 旧人体结果”的错帧叠加。
        if frame_id is None:
            self._frame_id += 1
        else:
            self._frame_id = int(frame_id)
        ts = int(time.time() * 1e6) if timestamp_us is None else int(timestamp_us)
        ts_lo, ts_hi = _split_u64(ts)
        fid_lo, _    = _split_u64(self._frame_id)

        self.header[h_fid]   = fid_lo
        self.header[h_ts_lo] = ts_lo
        self.header[h_ts_hi] = ts_hi
        self.header[h_ready] = 1        # slot 数据就绪

        # 原子切换：读端下次读时会看到新 slot
        self.header[_H_ACTIVE_SLOT] = write_slot
        self.mm.flush()

    def close(self, unlink: bool = False) -> None:
        try:
            try:
                self.mm.flush()
            except Exception:
                pass
            # Release numpy views before closing mmap, otherwise Python may raise:
            # BufferError: cannot close exported pointers exist
            try:
                self.header = None
                self._slots = []
            except Exception:
                pass
            try:
                self.mm.close()
            except Exception:
                pass
            try:
                os.close(self.fd)
            except Exception:
                pass
        finally:
            if unlink:
                try:
                    os.remove(self.path)
                except FileNotFoundError:
                    pass


class SharedFrameReader:
    """
    读端：只读 active_slot 指向的 slot。
    通过"读前/读后确认 active_slot 一致"来保证不读到撕裂数据。
    最多重试 _MAX_RETRIES 次，超过则返回 None。
    """

    _MAX_RETRIES = 4

    def __init__(self, name: str):
        self.name = name
        self.path = _backend_path(name)

        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)

        self.fd = os.open(self.path, os.O_RDONLY)
        self.total_bytes = os.path.getsize(self.path)
        self.mm = mmap.mmap(self.fd, self.total_bytes, access=mmap.ACCESS_READ)

        self.header = np.ndarray(
            (HEADER_U32_COUNT,), dtype=np.uint32, buffer=self.mm, offset=0
        )

        if int(self.header[_H_MAGIC]) != MAGIC:
            raise RuntimeError(
                "共享映射文件 magic 不匹配（期望双缓冲版 0x46524D32），"
                "可能是旧版单缓冲文件，请重启写端进程"
            )

        self.width    = int(self.header[_H_WIDTH])
        self.height   = int(self.header[_H_HEIGHT])
        self.channels = int(self.header[_H_CHANNELS])
        self.data_bytes = int(self.header[_H_DATA_BYTES])

        self._slots = [
            np.ndarray(
                (self.height, self.width, self.channels),
                dtype=np.uint8,
                buffer=self.mm,
                offset=HEADER_BYTES + i * self.data_bytes,
            )
            for i in range(2)
        ]
        self.last_frame_id = -1

    def read_latest(self) -> Optional[dict]:
        """
        返回最新帧字典，或 None（无新帧 / 读取竞争）。
        字典键：frame, frame_id, timestamp_us, shape
        """
        for _ in range(self._MAX_RETRIES):
            slot_a = int(self.header[_H_ACTIVE_SLOT])
            h_ready, h_fid, h_ts_lo, h_ts_hi = _SLOT_OFFSETS[slot_a]

            if int(self.header[h_ready]) != 1:
                return None

            frame_id = int(self.header[h_fid])
            if frame_id == self.last_frame_id:
                return None  # 没有新帧

            ts_lo = int(self.header[h_ts_lo])
            ts_hi = int(self.header[h_ts_hi])

            # 拷贝像素（关键：先于第二次 slot 读取）
            frame = self._slots[slot_a].copy()

            # 确认 active_slot 在读取过程中没有被写端切换走
            slot_b = int(self.header[_H_ACTIVE_SLOT])
            if slot_a != slot_b:
                # 写端刚好切换了 slot，重试
                continue

            self.last_frame_id = frame_id
            return {
                "frame":        frame,
                "frame_id":     frame_id,
                "timestamp_us": _join_u64(ts_lo, ts_hi),
                "shape":        (self.height, self.width, self.channels),
            }

        return None  # 重试耗尽，极罕见

    def close(self) -> None:
        try:
            try:
                self.header = None
                self._slots = []
            except Exception:
                pass
            self.mm.close()
        finally:
            os.close(self.fd)
