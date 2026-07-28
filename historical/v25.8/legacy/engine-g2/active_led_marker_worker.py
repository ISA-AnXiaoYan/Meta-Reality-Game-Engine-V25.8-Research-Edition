from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .marker_blink_decoder import (
        EVENT_DTYPE,
        EventPulseExtractor,
        MarkerBlinkDecoder,
        Observation,
        canonical_config_hash,
        decoder_config_dict,
        parse_roi,
    )
except ImportError:
    from marker_blink_decoder import (  # type: ignore
        EVENT_DTYPE,
        EventPulseExtractor,
        MarkerBlinkDecoder,
        Observation,
        canonical_config_hash,
        decoder_config_dict,
        parse_roi,
    )


Roi = Tuple[int, int, int, int]


class MarkerRoiTracker:
    def __init__(
        self,
        roi: Roi,
        *,
        frame_width: int = 0,
        frame_height: int = 0,
        search_margin_px: int = 12,
    ):
        self.base_roi = tuple(int(v) for v in roi)
        self.frame_width = max(0, int(frame_width))
        self.frame_height = max(0, int(frame_height))
        self.search_margin_px = max(0, int(search_margin_px))
        self._lock = threading.Lock()
        self._center = self._base_center()

    def _base_center(self) -> Tuple[float, float]:
        x, y, w, h = self.base_roi
        return float(x) + float(w) * 0.5, float(y) + float(h) * 0.5

    def update(self, x: float, y: float) -> None:
        with self._lock:
            self._center = (float(x), float(y))

    def reset(self) -> None:
        with self._lock:
            self._center = self._base_center()

    def capture_roi(self) -> Roi:
        with self._lock:
            cx, cy = self._center
        _, _, base_w, base_h = self.base_roi
        width = max(1, int(base_w) + 2 * self.search_margin_px)
        height = max(1, int(base_h) + 2 * self.search_margin_px)
        x = int(round(float(cx) - width * 0.5))
        y = int(round(float(cy) - height * 0.5))
        if self.frame_width > 0:
            x = max(0, min(x, max(0, self.frame_width - width)))
            width = min(width, self.frame_width)
        if self.frame_height > 0:
            y = max(0, min(y, max(0, self.frame_height - height)))
            height = min(height, self.frame_height)
        return int(x), int(y), int(width), int(height)

    def status(self) -> Dict[str, object]:
        with self._lock:
            center = [round(float(self._center[0]), 3), round(float(self._center[1]), 3)]
        return {
            "base_roi": list(self.base_roi),
            "capture_roi": list(self.capture_roi()),
            "center": center,
            "search_margin_px": int(self.search_margin_px),
        }


class ActiveLedMarkerSidecar:
    """Non-blocking active-marker decoder backend for the event worker."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        enabled: bool,
        camera: str,
        camera_serial: str,
        roi: Union[str, Sequence[int], Roi],
        output_path: str,
        status_path: str,
        perf_csv_path: str = "",
        run_id: str = "",
        frame_width: int = 0,
        frame_height: int = 0,
        queue_size: int = 512,
        search_margin_px: int = 12,
        stale_after_ms: float = 1000.0,
        status_interval_ms: float = 250.0,
        perf_interval_ms: float = 1000.0,
        alpha_ms: float = 20.0,
        bit_count: int = 8,
        checksum: bool = False,
        tolerance: float = 0.35,
        bin_us: int = 1000,
        min_events_per_bin: int = 8,
        min_segment_events: int = 20,
        merge_gap_us: int = 2500,
        polarity: str = "any",
        append_output: bool = False,
    ):
        self.enabled = bool(enabled)
        self.camera = str(camera)
        self.camera_serial = str(camera_serial or "")
        self.run_id = str(run_id or "")
        if isinstance(roi, str):
            parsed_roi = parse_roi(roi)
        else:
            values = tuple(int(v) for v in roi)
            parsed_roi = values if len(values) == 4 else None
        if self.enabled and parsed_roi is None:
            raise ValueError("active LED marker requires a bounded x,y,w,h ROI")
        self.roi: Roi = parsed_roi or (0, 0, 1, 1)
        self.output_path = Path(output_path)
        self.status_path = Path(status_path)
        self.perf_csv_path = Path(perf_csv_path) if str(perf_csv_path or "").strip() else None
        self.append_output = bool(append_output)
        self.stale_after_ms = max(50.0, float(stale_after_ms))
        self.status_interval_s = max(0.05, float(status_interval_ms) / 1000.0)
        self.perf_interval_s = max(0.1, float(perf_interval_ms) / 1000.0)
        self.alpha_us = int(round(float(alpha_ms) * 1000.0))
        self.bit_count = int(bit_count)
        self.checksum = bool(checksum)
        self.tolerance = float(tolerance)
        self.bin_us = int(bin_us)
        self.min_events_per_bin = int(min_events_per_bin)
        self.min_segment_events = int(min_segment_events)
        self.merge_gap_us = int(merge_gap_us)
        self.polarity = str(polarity)
        self.config = decoder_config_dict(
            alpha_us=self.alpha_us,
            bit_count=self.bit_count,
            checksum=self.checksum,
            tolerance=self.tolerance,
            bin_us=self.bin_us,
            min_events_per_bin=self.min_events_per_bin,
            min_segment_events=self.min_segment_events,
            merge_gap_us=self.merge_gap_us,
            polarity=self.polarity,
            roi=self.roi,
        )
        self.config_hash = canonical_config_hash(self.config)
        self.tracker = MarkerRoiTracker(
            self.roi,
            frame_width=frame_width,
            frame_height=frame_height,
            search_margin_px=search_margin_px,
        )
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._lock = threading.Lock()
        self._stats: Counter = Counter()
        self._last_error = ""
        self._last_marker_id: Optional[int] = None
        self._last_event_ts_us: Optional[int] = None
        self._last_wall_time_us: Optional[int] = None
        self._last_observation_mono: Optional[float] = None
        self._last_written_key: Optional[Tuple[str, int, int]] = None
        self._closed = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = "disabled" if not self.enabled else "starting"
        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            if self.perf_csv_path is not None:
                self.perf_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"ActiveLedMarker-{self.camera}",
            )
            self._thread.start()

    def submit(
        self,
        evs: np.ndarray,
        ts_us: int,
        wall_us: int,
        sync_meta: Optional[Dict[str, object]] = None,
    ) -> bool:
        if not self.enabled or self._closed:
            return False
        arr = np.asarray(evs)
        source_count = int(len(arr))
        capture_roi = self.tracker.capture_roi()
        if source_count:
            x, y, w, h = capture_roi
            mask = (
                (arr["x"] >= x)
                & (arr["x"] < x + w)
                & (arr["y"] >= y)
                & (arr["y"] < y + h)
            )
            selected = np.ascontiguousarray(arr[mask])
        else:
            selected = np.asarray([], dtype=arr.dtype if arr.dtype.fields else EVENT_DTYPE)
        item = (
            selected,
            int(ts_us or 0),
            int(wall_us or 0),
            dict(sync_meta or {}),
            capture_roi,
        )
        with self._lock:
            self._stats["batches_submitted"] += 1
            self._stats["source_events_seen"] += source_count
            self._stats["events_submitted"] += int(len(selected))
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._stats["queue_dropped"] += 1
                self._stats["events_dropped"] += int(len(selected))
            return False

    def _write_observation(
        self,
        fp,
        obs: Observation,
        *,
        wall_us: int,
        sync_meta: Dict[str, object],
    ) -> bool:
        key = (self.camera, int(obs.marker_id), int(obs.ts_us))
        if key == self._last_written_key:
            with self._lock:
                self._stats["observations_deduplicated"] += 1
            return False
        event_local = sync_meta.get("event_local_sync_index")
        mapped_mvs = sync_meta.get("mapped_mvs_sync_index")
        row = obs.to_json(
            camera=self.camera,
            alpha_us=self.alpha_us,
            checksum=self.checksum,
            source="event_worker_tap",
            run_id=self.run_id,
            camera_serial=self.camera_serial,
            wall_time_us=int(wall_us) if int(wall_us) > 0 else None,
            event_local_sync_index=None if event_local is None else int(event_local),
            mapped_mvs_sync_index=None if mapped_mvs is None else int(mapped_mvs),
            roi=self.roi,
            decoder_config_hash=self.config_hash,
        )
        fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp.flush()
        self._last_written_key = key
        self.tracker.update(obs.x, obs.y)
        with self._lock:
            self._stats["observations_written"] += 1
            self._last_marker_id = int(obs.marker_id)
            self._last_event_ts_us = int(obs.ts_us)
            self._last_wall_time_us = int(wall_us) if int(wall_us) > 0 else None
            self._last_observation_mono = time.monotonic()
        return True

    def _run(self) -> None:
        extractor = EventPulseExtractor(
            bin_us=self.bin_us,
            min_events_per_bin=self.min_events_per_bin,
            min_segment_events=self.min_segment_events,
            merge_gap_us=self.merge_gap_us,
            polarity=self.polarity,
            roi=None,
        )
        decoder = MarkerBlinkDecoder(
            alpha_us=self.alpha_us,
            bit_count=self.bit_count,
            checksum=self.checksum,
            tolerance=self.tolerance,
        )
        process_ms = []
        last_status = 0.0
        last_perf = time.monotonic()
        last_meta: Dict[str, object] = {}
        last_wall_us = 0
        output_mode = "a" if self.append_output else "w"
        perf_fp = None
        perf_writer = None
        try:
            if self.perf_csv_path is not None:
                perf_fp = self.perf_csv_path.open("w", encoding="utf-8", newline="")
                perf_writer = csv.DictWriter(perf_fp, fieldnames=[
                    "wall_time_us", "batches_processed", "events_processed", "observations_written",
                    "process_ms_p50", "process_ms_p95", "process_ms_p99", "queue_depth", "queue_dropped",
                ])
                perf_writer.writeheader()
            with self.output_path.open(output_mode, encoding="utf-8", buffering=1) as out:
                self._state = "running"
                while True:
                    try:
                        item = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        if self._stop.is_set():
                            break
                        item = None
                    if item is self._SENTINEL:
                        break
                    if item is not None:
                        arr, ts_us, wall_us, sync_meta, capture_roi = item
                        last_meta = sync_meta
                        last_wall_us = int(wall_us)
                        t0 = time.perf_counter()
                        pulses = extractor.process(arr, current_time_us=int(ts_us))
                        observations = decoder.process_pulses(pulses)
                        for obs in observations:
                            self._write_observation(out, obs, wall_us=wall_us, sync_meta=sync_meta)
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        process_ms.append(float(dt_ms))
                        with self._lock:
                            self._stats["batches_processed"] += 1
                            self._stats["events_processed"] += int(len(arr))
                            self._stats["pulses"] += int(len(pulses))
                            self._stats["last_batch_ts_us"] = int(ts_us)
                    now = time.monotonic()
                    self._refresh_stale(now)
                    if perf_writer is not None and now - last_perf >= self.perf_interval_s:
                        self._write_perf(perf_writer, process_ms)
                        perf_fp.flush()
                        process_ms.clear()
                        last_perf = now
                    if now - last_status >= self.status_interval_s:
                        self._write_status()
                        last_status = now
                pulses = extractor.flush()
                observations = decoder.process_pulses(pulses) + decoder.flush()
                for obs in observations:
                    self._write_observation(out, obs, wall_us=last_wall_us, sync_meta=last_meta)
                with self._lock:
                    self._stats["pulses"] += int(len(pulses))
                    self._stats["extractor_out_of_order_bins"] = int(extractor.out_of_order_bins)
                if perf_writer is not None and process_ms:
                    self._write_perf(perf_writer, process_ms)
                    perf_fp.flush()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._stats["worker_exceptions"] += 1
            self._state = "error"
        finally:
            if perf_fp is not None:
                try:
                    perf_fp.close()
                except Exception:
                    pass
            if self._state != "error":
                self._state = "stopped"
            self._write_status()

    def _write_perf(self, writer: csv.DictWriter, values) -> None:
        arr = np.asarray(values or [0.0], dtype=np.float64)
        snapshot = self.status()
        writer.writerow({
            "wall_time_us": int(time.time() * 1_000_000),
            "batches_processed": int(snapshot.get("batches_processed", 0)),
            "events_processed": int(snapshot.get("events_processed", 0)),
            "observations_written": int(snapshot.get("observations_written", 0)),
            "process_ms_p50": round(float(np.percentile(arr, 50)), 6),
            "process_ms_p95": round(float(np.percentile(arr, 95)), 6),
            "process_ms_p99": round(float(np.percentile(arr, 99)), 6),
            "queue_depth": int(snapshot.get("queue_depth", 0)),
            "queue_dropped": int(snapshot.get("queue_dropped", 0)),
        })

    def _refresh_stale(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            last = self._last_observation_mono
        if last is not None and (now - float(last)) * 1000.0 > self.stale_after_ms:
            self.tracker.reset()

    def status(self) -> Dict[str, object]:
        with self._lock:
            stats = dict(self._stats)
            last_obs = self._last_observation_mono
            last_marker = self._last_marker_id
            last_event = self._last_event_ts_us
            last_wall = self._last_wall_time_us
            last_error = self._last_error
        age_ms = None if last_obs is None else max(0.0, (time.monotonic() - float(last_obs)) * 1000.0)
        stale = bool(age_ms is not None and age_ms > self.stale_after_ms)
        state = self._state
        if self.enabled and state == "running" and stale:
            state = "stale"
        return {
            "enabled": bool(self.enabled),
            "state": state,
            "camera": self.camera,
            "camera_serial": self.camera_serial,
            "run_id": self.run_id,
            "last_event_ts_us": last_event,
            "last_wall_time_us": last_wall,
            "last_marker_id": last_marker,
            "last_observation_age_ms": None if age_ms is None else round(float(age_ms), 3),
            "stale": stale,
            "stale_after_ms": round(float(self.stale_after_ms), 3),
            "queue_depth": int(self._queue.qsize()),
            "queue_limit": int(self._queue.maxsize),
            "worker_alive": bool(self._thread is not None and self._thread.is_alive()),
            "last_error": last_error,
            "config_hash": self.config_hash,
            "roi_tracker": self.tracker.status(),
            **{str(k): int(v) for k, v in stats.items()},
        }

    def _write_status(self) -> None:
        if not self.enabled:
            return
        obj = self.status()
        tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.status_path)
        except Exception as exc:
            with self._lock:
                self._last_error = f"status_write:{type(exc).__name__}: {exc}"
                self._stats["status_write_errors"] += 1

    def close(self, timeout_sec: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        self._stop.set()
        try:
            self._queue.put(self._SENTINEL, timeout=max(0.1, float(timeout_sec) * 0.5))
        except queue.Full:
            with self._lock:
                self._last_error = "worker_stop_queue_timeout"
                self._stats["worker_stop_queue_timeout"] += 1
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, float(timeout_sec)))
            if self._thread.is_alive():
                with self._lock:
                    self._last_error = "worker_stop_timeout"
                    self._stats["worker_stop_timeout"] += 1
        self._write_status()


__all__ = ["ActiveLedMarkerSidecar", "MarkerRoiTracker"]
