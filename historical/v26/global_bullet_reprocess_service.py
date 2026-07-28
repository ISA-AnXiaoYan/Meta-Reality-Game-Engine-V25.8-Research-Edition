#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


SERVICE_NAME = "global_bullet_reprocess_service"
STATUS_VERSION = 1
RAW_FILES = {
    "bullet_points": "overlay_bullet_point_all.jsonl",
    "hit_candidates": "hit_candidate_all.jsonl",
    "global_person_tracks": "global_person_tracks.jsonl",
    "bev_overlay": "global_bev_overlay_record.jsonl",
}


def now_us() -> int:
    return int(time.time() * 1_000_000)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def count_jsonl(path: Path, max_scan: int = 100_000) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "line_count": 0, "valid_json_count": 0, "invalid_json_count": 0}
    line_count = 0
    valid = 0
    invalid = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            if line_count > max_scan:
                break
            try:
                json.loads(line)
                valid += 1
            except Exception:
                invalid += 1
    return {
        "exists": True,
        "line_count": line_count,
        "valid_json_count": valid,
        "invalid_json_count": invalid,
        "truncated_scan": line_count > max_scan,
    }


def load_calib_manifest(project_root: Path, explicit_path: str | None) -> dict[str, Any]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            project_root / "sync_ipc" / "calib_manifest.json",
            project_root / "calib_true" / "calib_manifest.json",
            project_root / "ids_8cam_fusion_config.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return {
                "path": str(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
            }
        except Exception as exc:
            return {"path": str(path), "exists": True, "error": repr(exc)}
    return {"exists": False, "path": None}


def point_line_distance(xy: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = xy
    dx = bx - ax
    dy = by - ay
    denom = math.hypot(dx, dy)
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / denom


def build_empty_decision(run_id: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "service": SERVICE_NAME,
        "decision_authority": "offline_diagnostic_only",
        "mode": "shadow",
        "status": "idle",
        "probable_hit_count": 0,
        "global_track_count": 0,
        "reason": reason,
        "ts_wall_us": now_us(),
    }


class Service:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.sync_root = Path(args.sync_root)
        if not self.sync_root.is_absolute():
            self.sync_root = self.project_root / self.sync_root
        self.output_dir = Path(args.output_dir) if args.output_dir else self.sync_root / "global_bullet_reprocess"
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        self.run_id = args.run_id or time.strftime("gbr_%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / "runs" / self.run_id
        self.stop_requested = False
        self.loop_count = 0
        self.started_us = now_us()

    def request_stop(self, *_: Any) -> None:
        self.stop_requested = True

    def raw_file_status(self) -> dict[str, Any]:
        return {
            key: {
                "path": str(self.sync_root / filename),
                **count_jsonl(self.sync_root / filename, self.args.max_scan_lines),
            }
            for key, filename in RAW_FILES.items()
        }

    def build_status(self, phase: str) -> dict[str, Any]:
        raw_status = self.raw_file_status()
        raw_pack_ready = all(raw_status[key]["exists"] and raw_status[key]["line_count"] > 0 for key in ("bullet_points", "hit_candidates", "global_person_tracks"))
        return {
            "schema_version": STATUS_VERSION,
            "service": SERVICE_NAME,
            "run_id": self.run_id,
            "phase": phase,
            "mode": self.args.mode,
            "decision_authority": "offline_diagnostic_only",
            "publish_to_official_hit_judge": False,
            "project_root": str(self.project_root),
            "sync_root": str(self.sync_root),
            "output_dir": str(self.output_dir),
            "run_output_dir": str(self.run_output_dir),
            "ts_wall_us": now_us(),
            "uptime_sec": round((now_us() - self.started_us) / 1_000_000.0, 3),
            "loop_count": self.loop_count,
            "raw_replay_status": raw_status,
            "raw_replay_pack_ready": raw_pack_ready,
            "calib_manifest": load_calib_manifest(self.project_root, self.args.calib_manifest),
            "empty_input_ok": bool(self.args.allow_empty),
            "remote_test_safe": True,
        }

    def write_outputs(self, status: dict[str, Any]) -> None:
        safe_mkdir(self.output_dir)
        safe_mkdir(self.run_output_dir)
        write_json_atomic(self.sync_root / "global_bullet_reprocess_status.json", status)
        write_json_atomic(self.sync_root / "global_bullet_reprocess_latest.json", status)
        write_json_atomic(self.output_dir / "status_latest.json", status)
        append_jsonl(self.output_dir / "status_history.jsonl", status)
        write_json_atomic(self.run_output_dir / "status_latest.json", status)
        append_jsonl(self.run_output_dir / "status_history.jsonl", status)
        if not status["raw_replay_pack_ready"]:
            append_jsonl(
                self.run_output_dir / "global_hit_decisions.jsonl",
                build_empty_decision(self.run_id, "raw_replay_pack_missing_or_empty"),
            )

    def run_once(self, phase: str) -> dict[str, Any]:
        status = self.build_status(phase)
        self.write_outputs(status)
        return status

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        safe_mkdir(self.output_dir)
        safe_mkdir(self.run_output_dir)
        duration_sec = float(self.args.duration_sec)
        end_time = None if duration_sec <= 0.0 else time.time() + duration_sec
        last_status: dict[str, Any] | None = None
        while not self.stop_requested and (end_time is None or time.time() <= end_time):
            self.loop_count += 1
            last_status = self.run_once("running")
            if self.args.once:
                break
            time.sleep(max(0.05, float(self.args.interval_sec)))
        self.loop_count += 1
        last_status = self.run_once("finished")
        report = {
            "run_id": self.run_id,
            "returncode": 0,
            "final_phase": last_status["phase"],
            "raw_replay_pack_ready": last_status["raw_replay_pack_ready"],
            "decision_authority": "offline_diagnostic_only",
            "publish_to_official_hit_judge": False,
            "status_path": str(self.sync_root / "global_bullet_reprocess_status.json"),
            "output_dir": str(self.output_dir),
            "run_output_dir": str(self.run_output_dir),
        }
        write_json_atomic(self.output_dir / "run_report.json", report)
        write_json_atomic(self.run_output_dir / "run_report.json", report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global bullet reprocess sidecar service in diagnostic/shadow mode.")
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--mode", choices=("capture_only", "shadow", "idle"), default="shadow")
    ap.add_argument("--duration-sec", type=float, default=10.0)
    ap.add_argument("--interval-sec", type=float, default=1.0)
    ap.add_argument("--allow-empty", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--max-scan-lines", type=int, default=100_000)
    ap.add_argument("--calib-manifest", default="")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.allow_empty and args.mode == "idle":
        args.allow_empty = True
    service = Service(args)
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
