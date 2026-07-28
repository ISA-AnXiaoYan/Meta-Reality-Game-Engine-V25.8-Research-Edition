from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
EVP_HEADER = struct.Struct("<4sHHQQQII")
EVP_MAGIC = b"EVP1"
PROTOCOL_VERSION = "ir-id-blink-v1"
OBSERVATION_SCHEMA_VERSION = 2


def canonical_config_hash(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def decoder_config_dict(
    *,
    alpha_us: int,
    bit_count: int,
    checksum: bool,
    tolerance: float,
    bin_us: int,
    min_events_per_bin: int,
    min_segment_events: int,
    merge_gap_us: int,
    polarity: str,
    roi: Optional[Tuple[int, int, int, int]],
) -> Dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "alpha_us": int(alpha_us),
        "bit_count": int(bit_count),
        "checksum": bool(checksum),
        "tolerance": round(float(tolerance), 8),
        "bin_us": int(bin_us),
        "min_events_per_bin": int(min_events_per_bin),
        "min_segment_events": int(min_segment_events),
        "merge_gap_us": int(merge_gap_us),
        "polarity": str(polarity),
        "roi": list(roi) if roi is not None else None,
    }


@dataclass
class Pulse:
    ts_us: int
    x: float
    y: float
    event_count: int
    bin_count: int


@dataclass
class Observation:
    ts_us: int
    marker_id: int
    confidence: float
    x: float
    y: float
    intervals_us: List[int]
    timing_error_ratio: float
    pulse_event_count: int
    pulse_count: int

    def to_json(
        self,
        *,
        camera: str,
        alpha_us: int,
        checksum: bool,
        source: str,
        run_id: str = "",
        camera_serial: str = "",
        wall_time_us: Optional[int] = None,
        event_local_sync_index: Optional[int] = None,
        mapped_mvs_sync_index: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        decoder_config_hash: str = "",
    ) -> Dict[str, object]:
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "msg_type": "active_marker_observation",
            "source": source,
            "run_id": str(run_id or ""),
            "camera": str(camera),
            "camera_serial": str(camera_serial or ""),
            "event_ts_us": int(self.ts_us),
            "wall_time_us": None if wall_time_us is None else int(wall_time_us),
            "event_local_sync_index": None if event_local_sync_index is None else int(event_local_sync_index),
            "mapped_mvs_sync_index": None if mapped_mvs_sync_index is None else int(mapped_mvs_sync_index),
            "ts_us": int(self.ts_us),
            "marker_id": int(self.marker_id),
            "confidence": round(float(self.confidence), 6),
            "event_x": round(float(self.x), 3),
            "event_y": round(float(self.y), 3),
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "roi": list(roi) if roi is not None else None,
            "alpha_ms": round(float(alpha_us) / 1000.0, 3),
            "checksum_enabled": bool(checksum),
            "pulse_count": int(self.pulse_count),
            "pulse_event_count": int(self.pulse_event_count),
            "timing_error_ratio": round(float(self.timing_error_ratio), 6),
            "intervals_us": [int(v) for v in self.intervals_us],
            "decoder_config_hash": str(decoder_config_hash or ""),
            "stale": False,
        }


@dataclass
class _BinAccumulator:
    bin_us: int
    event_count: int = 0
    x_sum: float = 0.0
    y_sum: float = 0.0
    t_sum: float = 0.0

    def add(self, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> None:
        self.event_count += int(len(ts))
        if len(ts):
            self.x_sum += float(np.sum(xs))
            self.y_sum += float(np.sum(ys))
            self.t_sum += float(np.sum(ts))


@dataclass
class _Segment:
    start_bin_us: int
    end_bin_us: int
    event_count: int = 0
    bin_count: int = 0
    x_sum: float = 0.0
    y_sum: float = 0.0
    t_sum: float = 0.0

    def add_bin(self, bin_us: int, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> None:
        count = int(len(ts))
        self.end_bin_us = int(bin_us)
        self.event_count += count
        self.bin_count += 1
        if count:
            self.x_sum += float(np.sum(xs))
            self.y_sum += float(np.sum(ys))
            self.t_sum += float(np.sum(ts))

    def add_summary(self, acc: _BinAccumulator) -> None:
        self.end_bin_us = int(acc.bin_us)
        self.event_count += int(acc.event_count)
        self.bin_count += 1
        self.x_sum += float(acc.x_sum)
        self.y_sum += float(acc.y_sum)
        self.t_sum += float(acc.t_sum)

    def to_pulse(self) -> Pulse:
        count = max(1, int(self.event_count))
        return Pulse(
            ts_us=int(round(self.t_sum / count)) if self.t_sum else int((self.start_bin_us + self.end_bin_us) // 2),
            x=float(self.x_sum / count) if self.x_sum else 0.0,
            y=float(self.y_sum / count) if self.y_sum else 0.0,
            event_count=int(self.event_count),
            bin_count=int(self.bin_count),
        )


class EventPulseExtractor:
    def __init__(
        self,
        *,
        bin_us: int = 1000,
        min_events_per_bin: int = 8,
        min_segment_events: int = 20,
        merge_gap_us: int = 2500,
        polarity: str = "any",
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.bin_us = max(100, int(bin_us))
        self.min_events_per_bin = max(1, int(min_events_per_bin))
        self.min_segment_events = max(1, int(min_segment_events))
        self.merge_gap_us = max(0, int(merge_gap_us))
        self.polarity = str(polarity or "any").lower()
        self.roi = roi
        self._pending: Optional[_Segment] = None
        self._pending_bin: Optional[_BinAccumulator] = None
        self.out_of_order_bins = 0

    def _filter_events(self, evs: np.ndarray) -> np.ndarray:
        if evs is None or len(evs) <= 0:
            return np.asarray([], dtype=EVENT_DTYPE)
        arr = np.asarray(evs)
        mask = np.ones(len(arr), dtype=bool)
        if self.polarity in ("positive", "pos", "1"):
            mask &= arr["p"] > 0
        elif self.polarity in ("negative", "neg", "0", "-1"):
            mask &= arr["p"] <= 0
        if self.roi is not None:
            x, y, w, h = self.roi
            mask &= (arr["x"] >= x) & (arr["x"] < x + w) & (arr["y"] >= y) & (arr["y"] < y + h)
        return arr[mask]

    def process(self, evs: np.ndarray, *, current_time_us: Optional[int] = None) -> List[Pulse]:
        arr = self._filter_events(evs)
        if len(arr) <= 0:
            return self.advance_time(current_time_us) if current_time_us is not None else []
        bins = (arr["t"].astype(np.int64) // self.bin_us) * self.bin_us
        uniq, inv = np.unique(bins, return_inverse=True)
        pulses: List[Pulse] = []
        for idx, bin_us in enumerate(uniq):
            bin_mask = inv == idx
            xs = arr["x"][bin_mask].astype(np.float64)
            ys = arr["y"][bin_mask].astype(np.float64)
            ts = arr["t"][bin_mask].astype(np.float64)
            current_bin = int(bin_us)
            if self._pending_bin is None:
                self._pending_bin = _BinAccumulator(bin_us=current_bin)
            elif current_bin < int(self._pending_bin.bin_us):
                self.out_of_order_bins += 1
                continue
            elif current_bin > int(self._pending_bin.bin_us):
                pulses.extend(self._finalize_pending_bin())
                self._pending_bin = _BinAccumulator(bin_us=current_bin)
            self._pending_bin.add(xs, ys, ts)
        if current_time_us is not None:
            pulses.extend(self.advance_time(current_time_us))
        return pulses

    def advance_time(self, current_time_us: Optional[int]) -> List[Pulse]:
        if current_time_us is None:
            return []
        now_us = int(current_time_us)
        out: List[Pulse] = []
        if self._pending_bin is not None and now_us >= int(self._pending_bin.bin_us) + self.bin_us:
            out.extend(self._finalize_pending_bin())
        max_gap = self.bin_us + self.merge_gap_us
        if self._pending is not None and now_us - int(self._pending.end_bin_us) > max_gap:
            out.extend(self._close_pending())
        return out

    def _finalize_pending_bin(self) -> List[Pulse]:
        if self._pending_bin is None:
            return []
        acc = self._pending_bin
        self._pending_bin = None
        if int(acc.event_count) < self.min_events_per_bin:
            return []
        return self._add_active_bin(acc)

    def _add_active_bin(self, acc: _BinAccumulator) -> List[Pulse]:
        out: List[Pulse] = []
        bin_us = int(acc.bin_us)
        if self._pending is None:
            self._pending = _Segment(start_bin_us=bin_us, end_bin_us=bin_us)
            self._pending.add_summary(acc)
            return out
        max_gap = self.bin_us + self.merge_gap_us
        if int(bin_us) - int(self._pending.end_bin_us) <= max_gap:
            self._pending.add_summary(acc)
            return out
        out.extend(self._close_pending())
        self._pending = _Segment(start_bin_us=bin_us, end_bin_us=bin_us)
        self._pending.add_summary(acc)
        return out

    def _close_pending(self) -> List[Pulse]:
        if self._pending is None:
            return []
        seg = self._pending
        self._pending = None
        if int(seg.event_count) < self.min_segment_events:
            return []
        return [seg.to_pulse()]

    def flush(self) -> List[Pulse]:
        return self._finalize_pending_bin() + self._close_pending()


class MarkerBlinkDecoder:
    def __init__(
        self,
        *,
        alpha_us: int = 20000,
        bit_count: int = 8,
        checksum: bool = False,
        tolerance: float = 0.35,
    ):
        self.alpha_us = max(1000, int(alpha_us))
        self.bit_count = max(1, min(16, int(bit_count)))
        self.checksum = bool(checksum)
        self.checksum_bits = 4 if self.checksum else 0
        self.tolerance = max(0.05, float(tolerance))
        self._pulses: Deque[Pulse] = deque(maxlen=256)
        self._last_emit_end_ts = -1

    def process_pulses(self, pulses: Sequence[Pulse]) -> List[Observation]:
        for pulse in pulses:
            if not self._pulses or int(pulse.ts_us) > int(self._pulses[-1].ts_us):
                self._pulses.append(pulse)
        return self._decode_available()

    def flush(self) -> List[Observation]:
        return self._decode_available()

    def _classify_start(self, dt_us: int) -> Tuple[bool, float]:
        return self._classify_nominal(dt_us, 3.0)

    def _classify_bit(self, dt_us: int) -> Tuple[Optional[int], float]:
        ok0, err0 = self._classify_nominal(dt_us, 1.0)
        ok1, err1 = self._classify_nominal(dt_us, 2.0)
        if ok0 and (not ok1 or err0 <= err1):
            return 0, err0
        if ok1:
            return 1, err1
        return None, min(err0, err1)

    def _classify_nominal(self, dt_us: int, slots: float) -> Tuple[bool, float]:
        ratio = float(dt_us) / float(self.alpha_us)
        err = abs(ratio - slots)
        return err <= self.tolerance, err

    def _decode_available(self) -> List[Observation]:
        pulses = list(self._pulses)
        needed_intervals = 1 + self.bit_count + self.checksum_bits
        if len(pulses) < needed_intervals + 1:
            return []
        intervals = [int(pulses[i + 1].ts_us) - int(pulses[i].ts_us) for i in range(len(pulses) - 1)]
        observations: List[Observation] = []
        for start_i in range(0, len(intervals) - needed_intervals + 1):
            if int(pulses[start_i + needed_intervals].ts_us) <= self._last_emit_end_ts:
                continue
            is_start, start_err = self._classify_start(intervals[start_i])
            if not is_start:
                continue
            bits: List[int] = []
            errs = [start_err]
            ok = True
            bit_intervals = intervals[start_i + 1:start_i + 1 + self.bit_count]
            for dt in bit_intervals:
                bit, err = self._classify_bit(dt)
                if bit is None:
                    ok = False
                    break
                bits.append(int(bit))
                errs.append(float(err))
            if not ok:
                continue
            marker_id = 0
            for bit in bits:
                marker_id = (marker_id << 1) | int(bit)
            check_bits: List[int] = []
            if self.checksum:
                check_intervals = intervals[
                    start_i + 1 + self.bit_count:start_i + 1 + self.bit_count + self.checksum_bits
                ]
                for dt in check_intervals:
                    bit, err = self._classify_bit(dt)
                    if bit is None:
                        ok = False
                        break
                    check_bits.append(int(bit))
                    errs.append(float(err))
                if not ok:
                    continue
                got = 0
                for bit in check_bits:
                    got = (got << 1) | int(bit)
                expected = ((marker_id >> 4) ^ (marker_id & 0x0F)) & 0x0F
                if got != expected:
                    continue
            end_pulse_i = start_i + needed_intervals
            used_pulses = pulses[start_i + 1:end_pulse_i + 1]
            event_count = sum(int(p.event_count) for p in used_pulses)
            avg_x = sum(float(p.x) for p in used_pulses) / float(max(1, len(used_pulses)))
            avg_y = sum(float(p.y) for p in used_pulses) / float(max(1, len(used_pulses)))
            timing_error = float(sum(errs)) / float(max(1, len(errs)))
            confidence = max(0.0, min(1.0, 1.0 - timing_error / max(self.tolerance, 1e-6)))
            observations.append(
                Observation(
                    ts_us=int(pulses[end_pulse_i].ts_us),
                    marker_id=int(marker_id),
                    confidence=float(confidence),
                    x=float(avg_x),
                    y=float(avg_y),
                    intervals_us=[int(v) for v in intervals[start_i:start_i + needed_intervals]],
                    timing_error_ratio=float(timing_error),
                    pulse_event_count=int(event_count),
                    pulse_count=int(len(used_pulses)),
                )
            )
            self._last_emit_end_ts = int(pulses[end_pulse_i].ts_us)
        self._trim_old()
        return observations

    def _trim_old(self) -> None:
        if self._last_emit_end_ts <= 0:
            return
        keep_after = self._last_emit_end_ts - int(5 * self.alpha_us)
        while len(self._pulses) > 2 and int(self._pulses[0].ts_us) < keep_after:
            self._pulses.popleft()


def parse_roi(value: str) -> Optional[Tuple[int, int, int, int]]:
    value = str(value or "").strip()
    if not value:
        return None
    parts = [int(float(p.strip())) for p in value.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def dtype_from_manifest(path: Optional[str]) -> np.dtype:
    if not path:
        return EVENT_DTYPE
    manifest_path = Path(path)
    if not manifest_path.exists():
        return EVENT_DTYPE
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    descr = obj.get("event_dtype_descr") or []
    fields = []
    for entry in descr:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            fields.append((str(entry[0]), str(entry[1])))
    if not fields:
        return EVENT_DTYPE
    return np.dtype(fields)


def iter_evp(path: str, *, manifest: Optional[str] = None, limit_slices: int = 0) -> Iterator[np.ndarray]:
    dtype = dtype_from_manifest(manifest)
    input_path = Path(path)
    if not input_path.is_absolute() and manifest:
        manifest_relative = Path(manifest).resolve().parent / input_path
        if manifest_relative.exists():
            input_path = manifest_relative
    with input_path.open("rb") as fp:
        count = 0
        while True:
            header = fp.read(EVP_HEADER.size)
            if not header:
                break
            if len(header) != EVP_HEADER.size:
                raise RuntimeError(f"short EVP header at slice {count}: {len(header)}")
            magic, version, header_size, seq, slice_ts_us, wall_us, event_count, event_record_size = EVP_HEADER.unpack(header)
            if magic != EVP_MAGIC:
                raise RuntimeError(f"bad EVP magic at slice {count}: {magic!r}")
            if version != 1:
                raise RuntimeError(f"unsupported EVP version: {version}")
            if header_size > EVP_HEADER.size:
                fp.read(int(header_size) - EVP_HEADER.size)
            if int(event_record_size) != int(dtype.itemsize):
                dtype = EVENT_DTYPE if int(event_record_size) == EVENT_DTYPE.itemsize else dtype
            payload_size = int(event_count) * int(event_record_size)
            payload = fp.read(payload_size)
            if len(payload) != payload_size:
                raise RuntimeError(f"short EVP payload seq={seq}: {len(payload)} < {payload_size}")
            yield np.frombuffer(payload, dtype=dtype, count=int(event_count))
            count += 1
            if limit_slices and count >= int(limit_slices):
                break


def iter_raw(
    path: str,
    *,
    delta_t_us: int = 4000,
    start_ts_us: int = 0,
    max_duration_us: int = 0,
    limit_slices: int = 0,
) -> Iterator[np.ndarray]:
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"RAW input not found: {raw_path}")
    try:
        from metavision_core.event_io import EventsIterator
    except Exception as exc:
        raise RuntimeError(
            "Metavision Python SDK is required for --raw. Run this command in the main-project "
            "Metavision 4.6.2 environment."
        ) from exc
    kwargs: Dict[str, object] = {
        "input_path": str(raw_path),
        "start_ts": max(0, int(start_ts_us)),
        "delta_t": max(1, int(delta_t_us)),
    }
    if int(max_duration_us) > 0:
        kwargs["max_duration"] = int(max_duration_us)
    iterator = EventsIterator(**kwargs)
    count = 0
    try:
        for evs in iterator:
            yield np.asarray(evs)
            count += 1
            if limit_slices and count >= int(limit_slices):
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def iter_evb_fifo(path: str, *, duration_sec: float = 0.0) -> Iterator[np.ndarray]:
    from native_event_bridge.hal_event_pipe_iterator import HalEventFifoIterator

    iterator = HalEventFifoIterator(fifo_path=str(path))
    start = time.monotonic()
    try:
        for evs in iterator:
            yield evs
            if duration_sec > 0 and time.monotonic() - start >= duration_sec:
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def write_observations(
    batches: Iterable[np.ndarray],
    *,
    args: argparse.Namespace,
    source: str,
) -> Dict[str, object]:
    started = time.perf_counter()
    roi = parse_roi(args.roi)
    alpha_us = int(round(float(args.alpha_ms) * 1000.0))
    config = decoder_config_dict(
        alpha_us=alpha_us,
        bit_count=int(args.bit_count),
        checksum=bool(args.checksum),
        tolerance=float(args.tolerance),
        bin_us=int(args.bin_us),
        min_events_per_bin=int(args.min_events_per_bin),
        min_segment_events=int(args.min_segment_events),
        merge_gap_us=int(args.merge_gap_us),
        polarity=str(args.polarity),
        roi=roi,
    )
    config_hash = canonical_config_hash(config)
    extractor = EventPulseExtractor(
        bin_us=int(args.bin_us),
        min_events_per_bin=int(args.min_events_per_bin),
        min_segment_events=int(args.min_segment_events),
        merge_gap_us=int(args.merge_gap_us),
        polarity=str(args.polarity),
        roi=roi,
    )
    decoder = MarkerBlinkDecoder(
        alpha_us=alpha_us,
        bit_count=int(args.bit_count),
        checksum=bool(args.checksum),
        tolerance=float(args.tolerance),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    def write_rows(out, observations: Sequence[Observation]) -> None:
        for obs in observations:
            out.write(json.dumps(
                obs.to_json(
                    camera=str(args.camera),
                    alpha_us=alpha_us,
                    checksum=bool(args.checksum),
                    source=source,
                    run_id=str(getattr(args, "run_id", "") or ""),
                    camera_serial=str(getattr(args, "camera_serial", "") or ""),
                    roi=roi,
                    decoder_config_hash=config_hash,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n")

    with out_path.open("w", encoding="utf-8") as out:
        for evs in batches:
            arr = np.asarray(evs)
            stats["slices"] += 1
            stats["events"] += int(len(arr))
            stats["max_batch_events"] = max(int(stats["max_batch_events"]), int(len(arr)))
            if len(arr):
                first_ts = int(np.min(arr["t"]))
                last_ts = int(np.max(arr["t"]))
                if not stats.get("first_event_ts_us"):
                    stats["first_event_ts_us"] = first_ts
                stats["last_event_ts_us"] = last_ts
                stats["positive_events"] += int(np.count_nonzero(arr["p"] > 0))
                stats["negative_events"] += int(np.count_nonzero(arr["p"] <= 0))
            pulses = extractor.process(arr)
            stats["pulses"] += int(len(pulses))
            observations = decoder.process_pulses(pulses)
            stats["observations"] += int(len(observations))
            write_rows(out, observations)
        pulses = extractor.flush()
        stats["pulses"] += int(len(pulses))
        observations = decoder.process_pulses(pulses) + decoder.flush()
        stats["observations"] += int(len(observations))
        write_rows(out, observations)
    elapsed_s = max(1e-9, time.perf_counter() - started)
    stats["elapsed_ms"] = round(elapsed_s * 1000.0, 3)
    stats["events_per_sec"] = round(float(stats["events"]) / elapsed_s, 3)
    stats["extractor_out_of_order_bins"] = int(extractor.out_of_order_bins)
    result = dict(stats)
    result["protocol_version"] = PROTOCOL_VERSION
    result["schema_version"] = OBSERVATION_SCHEMA_VERSION
    result["decoder_config_hash"] = config_hash
    return result


def synthetic_events(marker_id: int, *, alpha_us: int, pulse_width_us: int, checksum: bool) -> np.ndarray:
    rng = np.random.default_rng(7)
    slots = [3]
    for bit in range(7, -1, -1):
        slots.append(2 if ((marker_id >> bit) & 1) else 1)
    if checksum:
        c = ((marker_id >> 4) ^ (marker_id & 0x0F)) & 0x0F
        for bit in range(3, -1, -1):
            slots.append(2 if ((c >> bit) & 1) else 1)
    pulse_times = [100000]
    for slot in slots:
        pulse_times.append(pulse_times[-1] + int(slot * alpha_us))
    rows = []
    for t0 in pulse_times:
        for _ in range(120):
            rows.append((
                int(rng.normal(512, 2)),
                int(rng.normal(284, 2)),
                1,
                int(t0 + rng.integers(0, max(1, pulse_width_us))),
            ))
    arr = np.asarray(rows, dtype=EVENT_DTYPE)
    arr.sort(order="t")
    return arr


def run_self_test(args: argparse.Namespace) -> int:
    alpha_us = int(round(float(args.alpha_ms) * 1000.0))
    pulse_width_us = int(round(float(args.pulse_width_ms) * 1000.0))
    evs = synthetic_events(int(args.expected_id), alpha_us=alpha_us, pulse_width_us=pulse_width_us, checksum=bool(args.checksum))
    tmp_out = Path(args.output or "_marker_selftest.jsonl")
    args.output = str(tmp_out)
    stats = write_observations([evs], args=args, source="synthetic_self_test")
    observed = []
    if tmp_out.exists():
        for line in tmp_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                observed.append(json.loads(line))
    ok = any(int(row.get("marker_id", -1)) == int(args.expected_id) for row in observed)
    print(json.dumps({"ok": ok, "stats": stats, "observed": observed}, ensure_ascii=False, indent=2))
    if str(tmp_out) == "_marker_selftest.jsonl":
        try:
            tmp_out.unlink()
        except OSError:
            pass
    return 0 if ok else 2


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PoC decoder for active LED marker blink IDs.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--raw", default="", help="Metavision RAW file read through EventsIterator.")
    src.add_argument("--evp", default="", help="Offline event_slices.evp file.")
    src.add_argument("--evb-fifo", default="", help="Live EVB FIFO path written by hal_event_fifo_broker.")
    ap.add_argument("--manifest", default="", help="Optional event_slices_manifest.json for EVP dtype.")
    ap.add_argument("--output", default="sync_ipc/active_marker_observation_A.jsonl")
    ap.add_argument("--camera", default="A")
    ap.add_argument("--camera-serial", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--alpha-ms", type=float, default=20.0)
    ap.add_argument("--pulse-width-ms", type=float, default=10.0, help="Only used by --self-test.")
    ap.add_argument("--bit-count", type=int, default=8)
    ap.add_argument("--checksum", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.35)
    ap.add_argument("--bin-us", type=int, default=1000)
    ap.add_argument("--min-events-per-bin", type=int, default=8)
    ap.add_argument("--min-segment-events", type=int, default=20)
    ap.add_argument("--merge-gap-us", type=int, default=2500)
    ap.add_argument("--polarity", default="any", choices=["any", "positive", "negative", "pos", "neg"])
    ap.add_argument("--roi", default="", help="Optional x,y,w,h event-space ROI.")
    ap.add_argument("--limit-slices", type=int, default=0)
    ap.add_argument("--raw-delta-t-us", type=int, default=4000)
    ap.add_argument("--start-ts-us", type=int, default=0)
    ap.add_argument("--max-duration-us", type=int, default=0)
    ap.add_argument("--duration-sec", type=float, default=0.0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--expected-id", type=int, default=7)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    if args.raw:
        stats = write_observations(
            iter_raw(
                args.raw,
                delta_t_us=int(args.raw_delta_t_us),
                start_ts_us=int(args.start_ts_us),
                max_duration_us=int(args.max_duration_us),
                limit_slices=int(args.limit_slices),
            ),
            args=args,
            source="raw_offline",
        )
    elif args.evp:
        stats = write_observations(
            iter_evp(args.evp, manifest=args.manifest, limit_slices=int(args.limit_slices)),
            args=args,
            source="evp_offline",
        )
    elif args.evb_fifo:
        stats = write_observations(
            iter_evb_fifo(args.evb_fifo, duration_sec=float(args.duration_sec)),
            args=args,
            source="evb_fifo_live",
        )
    else:
        raise SystemExit("Provide --raw, --evp, --evb-fifo, or --self-test.")
    print(json.dumps({"output": str(args.output), "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
