#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hik_rgb_yolo_seg_multicam_same_model_stableid_v5 import MvsSharedFramePublisher, now_us  # noqa: E402
from preview_gpu_shm import RawPreviewFrameWriter, RAW_PIXEL_BAYER8  # noqa: E402
from remote_ops.check_v25_dataset_manifest import CAMERAS, build_report  # noqa: E402


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as fp:
        fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_trigger_replay_status(
    sync_root: Path,
    cams: List[str],
    sync_index: int,
    replay_wall_us: int,
    original_tick_wall_us: int,
    fps: float,
    total_published: int,
) -> None:
    timeline_rec = {
        "msg_type": "mvs_soft_trigger_tick",
        "sync_master": "mvs_soft_trigger",
        "source_mode": "mvs_file_replay",
        "program_sync_index": int(sync_index),
        "trigger_seq": int(sync_index),
        "mvs_soft_trigger_est_wall_us": int(replay_wall_us),
        "mvs_soft_trigger_original_wall_us": int(original_tick_wall_us or 0),
        "coord_send_wall_us": int(replay_wall_us),
        "period_ns": int(round(1_000_000_000.0 / max(1.0, float(fps or 40.0)))),
        "period_ms": float(1000.0 / max(1.0, float(fps or 40.0))),
        "fps": float(fps or 40.0),
        "lead_ms": 0.0,
        "cams": list(cams),
        "ok_cams": len(cams),
        "total_cams": len(cams),
        "pid": int(os.getpid()),
    }
    _append_jsonl(sync_root / "global_trigger_timeline.jsonl", timeline_rec)
    _write_json_atomic(sync_root / "global_trigger_timeline_latest.json", timeline_rec)
    _write_json_atomic(sync_root / "trigger_status.json", {
        "state": "running",
        "source_mode": "mvs_file_replay",
        "tick_id": int(sync_index),
        "fps": float(fps or 40.0),
        "trigger_fps": float(fps or 40.0),
        "period_ms": float(1000.0 / max(1.0, float(fps or 40.0))),
        "lead_ms": 0.0,
        "ok_cams": len(cams),
        "total_cams": len(cams),
        "total_frames_published": int(total_published),
        "wall_time_us": int(replay_wall_us),
    })


def _write_sync_start_flag(
    sync_root: Path,
    cams: List[str],
    sync_index: int,
    replay_wall_us: int,
    fps: float,
    total_published: int,
) -> None:
    payload = {
        "state": "ready",
        "source_mode": "mvs_file_replay",
        "sync_master": "mvs_soft_trigger",
        "reason": "mvs_file_frame_broker_first_aligned_batch",
        "event_sync_direct_index_map": True,
        "event_sync_mapping": "sync_start_tick_id_plus_event_local_sync_index",
        "tick_id_start": int(sync_index),
        "program_sync_index_start": int(sync_index),
        "sync_index_start": int(sync_index),
        "wall_time_us": int(replay_wall_us),
        "fps": float(fps or 40.0),
        "period_ms": float(1000.0 / max(1.0, float(fps or 40.0))),
        "cams": list(cams),
        "ok_cams": len(cams),
        "total_cams": len(cams),
        "total_frames_published": int(total_published),
        "pid": int(os.getpid()),
    }
    _write_json_atomic(sync_root / "sync_start.flag", payload)
    _write_json_atomic(sync_root / "sync_start_latest.json", payload)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
        return list(csv.DictReader(fp))


class CameraReplay:
    def __init__(self, cam: str, video_path: Path, index_path: Path):
        self.cam = str(cam)
        self.video_path = video_path
        self.index_path = index_path
        self.rows = _read_csv_rows(index_path)
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open MVS replay video {cam}: {video_path}")
        self.frame_idx = 0
        self.frames_published = 0
        self.errors = 0
        self.last_error = ""
        self.finished = False
        self.raw_writer: Optional[RawPreviewFrameWriter] = None
        self.raw_shape: Optional[Tuple[int, int]] = None

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            if self.raw_writer is not None:
                self.raw_writer.close(unlink=True)
        except Exception:
            pass

    def read_next(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        if self.finished:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.finished = True
            return None
        row = self.rows[self.frame_idx] if self.frame_idx < len(self.rows) else {}
        self.frame_idx += 1
        local_fid = _safe_int(row.get("local_fid"), self.frame_idx)
        sync_index = _safe_int(row.get("program_sync_index"), max(0, local_fid - 1))
        mvs_frame_num = _safe_int(row.get("mvs_frame_num"), local_fid)
        capture_ts_us = _safe_int(row.get("capture_ts_us"), now_us())
        meta = {
            "program_sync_index": int(sync_index),
            "sync_index": int(sync_index),
            "sync_master": "mvs_soft_trigger",
            "soft_trigger_mode": "mvs_file_frame_broker",
            "mvs_source_mode": "file_replay",
            "mvs_file_replay_video_path": str(self.video_path),
            "mvs_file_replay_index_path": str(self.index_path),
            "mvs_file_replay_record_frame_idx": int(_safe_int(row.get("record_frame_idx"), self.frame_idx - 1)),
            "mvs_frame_num": int(mvs_frame_num),
            "capture_ts_us": int(capture_ts_us),
            "receiver_wall_us": int(now_us()),
            "record_wall_us_original": _safe_int(row.get("record_wall_us"), 0),
            "soft_trigger_tick_wall_us": _safe_int(row.get("soft_trigger_tick_wall_us"), 0),
            "soft_trigger_cmd_wall_us": _safe_int(row.get("soft_trigger_cmd_wall_us"), 0),
            "soft_trigger_cmd_spread_us": _safe_float(row.get("soft_trigger_cmd_spread_us"), 0.0),
            "rgb_exposure_as_sync_anchor": False,
        }
        return frame, {
            "fid": int(local_fid),
            "timestamp_us": int(capture_ts_us),
            "mvs_frame_num": int(mvs_frame_num),
            "sync_index": int(sync_index),
            "meta": meta,
            "row": row,
        }

    def ensure_raw_writer(self, gray: np.ndarray, name_template: str, slot_count: int) -> Optional[RawPreviewFrameWriter]:
        h, w = gray.shape[:2]
        shape = (int(h), int(w))
        if self.raw_writer is not None and self.raw_shape == shape:
            return self.raw_writer
        if self.raw_writer is not None:
            try:
                self.raw_writer.close(unlink=True)
            except Exception:
                pass
            self.raw_writer = None
        name = str(name_template).format(camera=self.cam, cam=self.cam, id=self.cam)
        self.raw_writer = RawPreviewFrameWriter(
            name,
            width=int(w),
            height=int(h),
            stride=int(w),
            pixel_format=RAW_PIXEL_BAYER8,
            bayer_pattern="BG",
            slot_count=int(slot_count),
            create=True,
            flush=False,
        )
        self.raw_shape = shape
        return self.raw_writer


def _build_sync_cfg(sync_root: Path) -> Dict[str, Any]:
    return {
        "root": str(sync_root),
        "mvs_frame_pub_enable": True,
        "mvs_frame_pub_name_template": "mvs_latest_{camera}",
        "mvs_frame_pub_meta_template": "{root}/mvs_latest_{camera}.json",
    }


def _select_cameras(requested: str) -> List[str]:
    if not requested:
        return list(CAMERAS)
    out = []
    for item in requested.replace(",", "").upper():
        if item in CAMERAS and item not in out:
            out.append(item)
    return out or list(CAMERAS)


def main() -> int:
    ap = argparse.ArgumentParser(description="V25-C MVS file frame broker: replay recorded MVS AVI/CSV to existing shm contracts.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--cams", default="ABCDEFGH")
    ap.add_argument("--replay-timing", choices=["realtime", "fast"], default="realtime")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--raw-preview-enable", action="store_true", default=True)
    ap.add_argument("--no-raw-preview", dest="raw_preview_enable", action="store_false")
    ap.add_argument("--raw-preview-template", default="mvs_raw_latest_{camera}")
    ap.add_argument("--raw-preview-slot-count", type=int, default=8)
    ap.add_argument("--status-json", default="sync_ipc/mvs_file_frame_broker_status.json")
    ap.add_argument("--stats-jsonl", default="sync_ipc/mvs_file_frame_broker_stats.jsonl")
    ap.add_argument("--report-ms", type=int, default=1000)
    ap.add_argument("--publish-trigger-timeline", action="store_true", default=True)
    ap.add_argument("--no-publish-trigger-timeline", dest="publish_trigger_timeline", action="store_false")
    ap.add_argument("--trigger-fps", type=float, default=40.0)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    report = build_report(dataset_dir, mode="mvs_replay")
    video_map = report.get("paths", {}).get("mvs_video_by_camera", {})
    index_map = report.get("paths", {}).get("mvs_frame_index_by_camera", {})
    if not isinstance(video_map, dict) or not isinstance(index_map, dict):
        raise SystemExit("dataset manifest report did not contain MVS maps")

    cams = _select_cameras(str(args.cams))
    runtimes: Dict[str, CameraReplay] = {}
    for cam in cams:
        if cam not in video_map or cam not in index_map:
            raise SystemExit(f"missing MVS replay input for camera {cam}")
        runtimes[cam] = CameraReplay(cam, Path(str(video_map[cam])), Path(str(index_map[cam])))

    stop = False

    def _handle_stop(signum=None, frame=None) -> None:
        nonlocal stop
        stop = True

    try:
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
    except Exception:
        pass

    sync_root = Path(args.sync_root).expanduser()
    if not sync_root.is_absolute():
        sync_root = Path.cwd() / sync_root
    publisher = MvsSharedFramePublisher(_build_sync_cfg(sync_root), cams)
    status_path = Path(args.status_json)
    stats_path = Path(args.stats_jsonl)
    report_interval_s = max(0.1, float(args.report_ms) / 1000.0)
    last_report = 0.0
    first_replay_sync: Optional[int] = None
    first_wall = time.monotonic()
    total_published = 0
    tick = 0
    last_trigger_sync_index: Optional[int] = None
    pending: Dict[str, Optional[Tuple[np.ndarray, Dict[str, Any]]]] = {cam: None for cam in cams}
    frames_published_total: Dict[str, int] = {cam: 0 for cam in cams}
    skipped_to_align: Dict[str, int] = {cam: 0 for cam in cams}
    sync_start_written = False
    trigger_timeline_publish_count = 0

    def _item_sync(item: Optional[Tuple[np.ndarray, Dict[str, Any]]]) -> int:
        if item is None:
            return -1
        try:
            return int((item[1] or {}).get("sync_index", -1))
        except Exception:
            return -1

    def _read_pending(cam: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        item = runtimes[cam].read_next()
        pending[cam] = item
        return item

    def _next_aligned_batch() -> Tuple[Optional[int], Dict[str, Tuple[np.ndarray, Dict[str, Any]]]]:
        for cam in cams:
            if pending.get(cam) is None:
                _read_pending(cam)
        while True:
            if any(pending.get(cam) is None for cam in cams):
                return None, {}
            syncs = {cam: _item_sync(pending.get(cam)) for cam in cams}
            target_sync = max(syncs.values())
            advanced = False
            for cam in cams:
                while pending.get(cam) is not None and _item_sync(pending.get(cam)) < target_sync:
                    skipped_to_align[cam] = int(skipped_to_align.get(cam, 0)) + 1
                    _read_pending(cam)
                    advanced = True
            if any(pending.get(cam) is None for cam in cams):
                return None, {}
            syncs = {cam: _item_sync(pending.get(cam)) for cam in cams}
            if len(set(syncs.values())) == 1:
                batch = {cam: pending[cam] for cam in cams if pending.get(cam) is not None}
                for cam in cams:
                    pending[cam] = None
                return int(next(iter(syncs.values()))), batch  # type: ignore[return-value]
            if not advanced:
                target_sync = max(syncs.values())
                for cam in cams:
                    if _item_sync(pending.get(cam)) < target_sync:
                        skipped_to_align[cam] = int(skipped_to_align.get(cam, 0)) + 1
                        _read_pending(cam)

    try:
        while not stop:
            sync_index, batch = _next_aligned_batch()
            if not batch or sync_index is None:
                active = 0
            else:
                active = len(batch)

            if active > 0 and args.replay_timing == "realtime":
                if first_replay_sync is None:
                    first_replay_sync = int(sync_index)
                    first_wall = time.monotonic()
                delay_s = max(0.0, (int(sync_index) - int(first_replay_sync)) / max(1.0, float(args.trigger_fps or 40.0)))
                target = first_wall + delay_s
                now = time.monotonic()
                if target > now:
                    time.sleep(min(0.05, target - now))

            batch_wall_us = int(now_us())
            for cam, item in batch.items():
                rt = runtimes[cam]
                frame, meta = item
                meta_dict = dict(meta.get("meta", {}) or {})
                original_capture_ts_us = int(meta_dict.get("capture_ts_us", meta.get("timestamp_us", 0)) or 0)
                meta_dict.update({
                    "capture_ts_us": int(batch_wall_us),
                    "timestamp_us": int(batch_wall_us),
                    "receiver_wall_us": int(batch_wall_us),
                    "replay_wall_us": int(batch_wall_us),
                    "mvs_soft_trigger_est_wall_us": int(batch_wall_us),
                    "soft_trigger_cmd_wall_us": int(batch_wall_us),
                    "soft_trigger_cmd_done_wall_us": int(batch_wall_us),
                    "rgb_exposure_time_us": 0,
                    "exposure_time_us": 0,
                    "rgb_exposure_source": "mvs_file_replay_zero_window",
                    "exposure_source": "mvs_file_replay_zero_window",
                    "rgb_exposure_start_offset_us": 0,
                    "rgb_exposure_mid_offset_us": 0,
                    "rgb_exposure_end_offset_us": 0,
                    "record_capture_ts_us_original": int(original_capture_ts_us),
                    "mvs_file_replay_aligned_batch": True,
                    "mvs_file_replay_skipped_to_align": int(skipped_to_align.get(cam, 0)),
                })
                publisher.write(
                    cam,
                    frame,
                    int(batch_wall_us),
                    int(meta.get("fid", rt.frame_idx)),
                    int(meta.get("mvs_frame_num", rt.frame_idx)),
                    meta_dict,
                )
                if bool(args.raw_preview_enable):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                    rw = rt.ensure_raw_writer(gray, str(args.raw_preview_template), int(args.raw_preview_slot_count))
                    if rw is not None:
                        rw.write(
                            gray,
                            int(batch_wall_us),
                            int(meta.get("fid", rt.frame_idx)),
                            tick_id=int(sync_index),
                            frame_recv_ns=int(batch_wall_us) * 1000,
                            trigger_fire_ns=int(batch_wall_us) * 1000,
                        )
                rt.frames_published += 1
                frames_published_total[cam] = int(frames_published_total.get(cam, 0)) + 1
                total_published += 1

            if active > 0 and bool(args.publish_trigger_timeline) and sync_index != last_trigger_sync_index:
                first_item = next(iter(batch.values()))
                original_tick_wall_us = int(((first_item[1].get("meta", {}) or {}).get("soft_trigger_tick_wall_us", 0)) or 0)
                if not sync_start_written:
                    _write_sync_start_flag(
                        sync_root,
                        cams,
                        int(sync_index),
                        int(batch_wall_us),
                        float(args.trigger_fps or 40.0),
                        int(total_published),
                    )
                    sync_start_written = True
                _write_trigger_replay_status(
                    sync_root,
                    cams,
                    int(sync_index),
                    int(batch_wall_us),
                    original_tick_wall_us,
                    float(args.trigger_fps or 40.0),
                    int(total_published),
                )
                trigger_timeline_publish_count += 1
                last_trigger_sync_index = int(sync_index)

            if active == 0:
                if args.loop:
                    for rt in runtimes.values():
                        rt.close()
                    runtimes = {
                        cam: CameraReplay(cam, Path(str(video_map[cam])), Path(str(index_map[cam])))
                        for cam in cams
                    }
                    pending = {cam: None for cam in cams}
                    first_replay_sync = None
                    last_trigger_sync_index = None
                    continue
                break
            if args.max_frames > 0 and total_published >= int(args.max_frames):
                break

            now_s = time.monotonic()
            if now_s - last_report >= report_interval_s:
                tick += 1
                payload = {
                    "schema": "mvs_file_frame_broker_status.v1",
                    "source_mode": "file_replay",
                    "dataset_dir": str(dataset_dir),
                    "tick": int(tick),
                    "published_wall_us": int(now_us()),
                    "replay_timing": str(args.replay_timing),
                    "raw_preview_enable": bool(args.raw_preview_enable),
                    "publish_trigger_timeline": bool(args.publish_trigger_timeline),
                    "sync_start_written": bool(sync_start_written),
                    "trigger_timeline_publish_count": int(trigger_timeline_publish_count),
                    "last_trigger_sync_index": None if last_trigger_sync_index is None else int(last_trigger_sync_index),
                    "cameras_requested": len(cams),
                    "total_frames_published": int(total_published),
                    "cameras": [
                        {
                            "camera": cam,
                            "video_path": str(rt.video_path),
                            "index_path": str(rt.index_path),
                            "frames_published": int(frames_published_total.get(cam, rt.frames_published)),
                            "finished": bool(rt.finished),
                            "errors": int(rt.errors),
                            "last_error": str(rt.last_error or ""),
                            "skipped_to_align": int(skipped_to_align.get(cam, 0)),
                        }
                        for cam, rt in sorted(runtimes.items())
                    ],
                }
                _write_json_atomic(status_path, payload)
                _append_jsonl(stats_path, payload)
                last_report = now_s
        tick += 1
        final_payload = {
            "schema": "mvs_file_frame_broker_status.v1",
            "source_mode": "file_replay",
            "dataset_dir": str(dataset_dir),
            "tick": int(tick),
            "published_wall_us": int(now_us()),
            "state": "stopped" if stop else "finished",
            "publish_trigger_timeline": bool(args.publish_trigger_timeline),
            "sync_start_written": bool(sync_start_written),
            "trigger_timeline_publish_count": int(trigger_timeline_publish_count),
            "last_trigger_sync_index": None if last_trigger_sync_index is None else int(last_trigger_sync_index),
            "total_frames_published": int(total_published),
            "cameras": [
                {
                    "camera": cam,
                    "frames_published": int(frames_published_total.get(cam, rt.frames_published)),
                    "finished": bool(rt.finished),
                    "errors": int(rt.errors),
                    "last_error": str(rt.last_error or ""),
                    "skipped_to_align": int(skipped_to_align.get(cam, 0)),
                }
                for cam, rt in sorted(runtimes.items())
            ],
        }
        _write_json_atomic(status_path, final_payload)
        _append_jsonl(stats_path, final_payload)
        print(json.dumps(final_payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        for rt in runtimes.values():
            rt.close()


if __name__ == "__main__":
    raise SystemExit(main())
