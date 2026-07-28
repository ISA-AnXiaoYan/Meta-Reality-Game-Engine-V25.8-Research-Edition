#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-latency human-mask record transport over /dev/shm mmap files.

The payload is UTF-8 JSON guarded by a tiny seqlock header.  Writers publish one
latest record per camera; readers accept only stable even sequence numbers.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MAGIC = b"HMASKV1\0"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIIQQ")
HEADER_SIZE = HEADER_STRUCT.size
DEFAULT_BYTES = 1024 * 1024


def now_us() -> int:
    return int(time.time() * 1_000_000)


def resolve_mask_path(template: str, camera: str) -> str:
    raw = str(template or "/dev/shm/event_human_mask_latest_{camera}.mmap")
    try:
        raw = raw.format(camera=camera, cam=camera, id=camera, alias=camera)
    except Exception:
        pass
    if raw.startswith("/"):
        return raw
    if raw.endswith(".mmap"):
        return os.path.join("/dev/shm", raw)
    return os.path.join("/dev/shm", raw + ".mmap")


class HumanMaskFastPublisher:
    def __init__(self, template: str = "/dev/shm/event_human_mask_latest_{camera}.mmap",
                 size_bytes: int = DEFAULT_BYTES):
        self.template = str(template or "/dev/shm/event_human_mask_latest_{camera}.mmap")
        self.size_bytes = max(HEADER_SIZE + 4096, int(size_bytes or DEFAULT_BYTES))
        self._items: Dict[str, Tuple[int, mmap.mmap, str]] = {}
        self._seq: Dict[str, int] = {}

    def _open(self, camera: str) -> Tuple[int, mmap.mmap, str]:
        cam = str(camera)
        item = self._items.get(cam)
        if item is not None:
            return item
        path = resolve_mask_path(self.template, cam)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, self.size_bytes)
        mm = mmap.mmap(fd, self.size_bytes, access=mmap.ACCESS_WRITE)
        item = (fd, mm, path)
        self._items[cam] = item
        return item

    def publish(self, camera: str, record: Dict[str, Any]) -> Tuple[bool, str, int]:
        cam = str(camera)
        fd, mm, path = self._open(cam)
        seq = int(self._seq.get(cam, 0)) + 2
        if seq % 2:
            seq += 1
        payload = dict(record or {})
        payload.setdefault("msg_type", "human_mask_fast_record")
        payload["mask_fast_seq"] = int(seq)
        payload["mask_fast_path"] = str(path)
        payload.setdefault("mask_publish_wall_us", now_us())
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        capacity = self.size_bytes - HEADER_SIZE
        if len(data) > capacity:
            return False, f"payload_too_large:{len(data)}>{capacity}", int(self._seq.get(cam, 0))
        odd_seq = seq - 1
        mm.seek(0)
        mm.write(HEADER_STRUCT.pack(MAGIC, VERSION, HEADER_SIZE, odd_seq, 0))
        mm.seek(HEADER_SIZE)
        mm.write(data)
        if len(data) < capacity:
            mm.write(b"\0")
        mm.seek(0)
        mm.write(HEADER_STRUCT.pack(MAGIC, VERSION, HEADER_SIZE, seq, len(data)))
        self._seq[cam] = seq
        return True, "", seq

    def close(self) -> None:
        for fd, mm, _path in list(self._items.values()):
            try:
                mm.close()
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass
        self._items.clear()


class HumanMaskFastReader:
    def __init__(self, template: str = "/dev/shm/event_human_mask_latest_{camera}.mmap",
                 camera: str = "", size_bytes: int = DEFAULT_BYTES):
        self.template = str(template or "/dev/shm/event_human_mask_latest_{camera}.mmap")
        self.camera = str(camera or "")
        self.size_bytes = max(HEADER_SIZE + 4096, int(size_bytes or DEFAULT_BYTES))
        self.path = resolve_mask_path(self.template, self.camera)
        self.fd: Optional[int] = None
        self.mm: Optional[mmap.mmap] = None
        self.last_seq = -1
        self.last_error = ""
        self.open_count = 0
        self.read_count = 0
        self.skip_count = 0

    def _open(self) -> bool:
        if self.mm is not None:
            return True
        try:
            self.fd = os.open(self.path, os.O_RDONLY)
            file_size = max(HEADER_SIZE, os.path.getsize(self.path))
            self.mm = mmap.mmap(self.fd, file_size, access=mmap.ACCESS_READ)
            self.open_count += 1
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            return False

    def read_latest(self, require_new: bool = True) -> Optional[Dict[str, Any]]:
        if not self._open() or self.mm is None:
            self.skip_count += 1
            return None
        try:
            self.mm.seek(0)
            h1 = self.mm.read(HEADER_SIZE)
            magic, version, header_size, seq1, payload_len1 = HEADER_STRUCT.unpack(h1)
            if magic != MAGIC or int(version) != VERSION or int(header_size) != HEADER_SIZE:
                self.last_error = "bad_header"
                self.skip_count += 1
                return None
            if int(seq1) <= 0 or int(seq1) % 2:
                self.last_error = "writer_busy"
                self.skip_count += 1
                return None
            if require_new and int(seq1) == int(self.last_seq):
                self.skip_count += 1
                return None
            payload_len = int(payload_len1)
            if payload_len <= 0 or payload_len > len(self.mm) - HEADER_SIZE:
                self.last_error = f"bad_payload_len:{payload_len}"
                self.skip_count += 1
                return None
            self.mm.seek(HEADER_SIZE)
            data = self.mm.read(payload_len)
            self.mm.seek(0)
            h2 = self.mm.read(HEADER_SIZE)
            _magic2, _version2, _header_size2, seq2, payload_len2 = HEADER_STRUCT.unpack(h2)
            if int(seq1) != int(seq2) or int(seq2) % 2 or int(payload_len1) != int(payload_len2):
                self.last_error = "unstable_read"
                self.skip_count += 1
                return None
            obj = json.loads(data.decode("utf-8"))
            if not isinstance(obj, dict):
                self.last_error = "payload_not_dict"
                self.skip_count += 1
                return None
            self.last_seq = int(seq2)
            self.read_count += 1
            self.last_error = ""
            obj["_mask_fast_read_wall_us"] = now_us()
            obj["_mask_fast_reader_seq"] = int(seq2)
            obj["_mask_fast_reader_path"] = self.path
            return obj
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.skip_count += 1
            return None

    def status(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "last_seq": int(self.last_seq),
            "open_count": int(self.open_count),
            "read_count": int(self.read_count),
            "skip_count": int(self.skip_count),
            "last_error": str(self.last_error),
        }

    def close(self) -> None:
        try:
            if self.mm is not None:
                self.mm.close()
        except Exception:
            pass
        try:
            if self.fd is not None:
                os.close(self.fd)
        except Exception:
            pass
        self.mm = None
        self.fd = None
