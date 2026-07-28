#!/usr/bin/env python3
"""Validate External Trigger events in recorded Prophesee RAW files.

This tool is intentionally a thin wrapper around native_event_bridge/event_file_fifo_broker
in --inspect-only mode so the validation path and replay path decode RAW files the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_broker_paths(project_root: Path) -> Iterable[Path]:
    exe = ".exe" if os.name == "nt" else ""
    yield project_root / "native_event_bridge" / "build" / f"event_file_fifo_broker{exe}"
    yield project_root / "native_event_bridge" / "build" / "Release" / f"event_file_fifo_broker{exe}"
    yield project_root / "native_event_bridge" / "build" / "Debug" / f"event_file_fifo_broker{exe}"


def _find_broker(project_root: Path, explicit: str = "") -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    for p in _candidate_broker_paths(project_root):
        if p.exists():
            return p
    return None


def _infer_alias(path: Path, used: set[str]) -> str:
    stem = path.stem.upper()
    patterns = [
        r"(?:^|[_-])EVENT[_-]?([A-H])(?:[_-]|$)",
        r"(?:^|[_-])CAMERA[_-]?([A-H])(?:[_-]|$)",
        r"(?:^|[_-])([A-H])(?:[_-]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            alias = m.group(1)
            if alias not in used:
                return alias
    for alias in "ABCDEFGH":
        if alias not in used:
            return alias
    return f"cam{len(used)}"


def _discover_raw_files(paths: List[str]) -> List[Tuple[str, Path]]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files.extend(sorted(p.rglob("*.raw")))
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(str(p))
    used: set[str] = set()
    out: List[Tuple[str, Path]] = []
    for p in files:
        alias = _infer_alias(p, used)
        used.add(alias)
        out.append((alias, p.resolve()))
    return out


def _count_text_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            return sum(1 for line in fp if line.strip())
    except Exception:
        return -1


def _discover_replay_sync_anchor_rows(paths: List[str]) -> Dict[str, int]:
    roots: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            roots.append(p)
        elif p.exists():
            roots.append(p.parent)
    out: Dict[str, int] = {}
    for root in roots:
        for path in sorted(root.rglob("event_trigger_map_*.jsonl")):
            alias = _infer_alias(path, set(out))
            if alias in out:
                continue
            out[alias] = _count_text_rows(path)
    return dict(sorted(out.items()))


def _parse_json_line(text: str) -> Dict[str, Any]:
    last_obj = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last_obj = line
    if not last_obj:
        raise RuntimeError("event_file_fifo_broker did not print a JSON result")
    return json.loads(last_obj)


def _run_broker_inspect(
    broker: Path,
    files: List[Tuple[str, Path]],
    *,
    stats_jsonl: Path,
    slice_us: int,
    replay_timing: str,
    allow_missing_ext_trigger: bool,
) -> Tuple[Dict[str, Any], int, str, str]:
    raw_files = ",".join(f"{alias}={path}" for alias, path in files)
    cmd = [
        str(broker),
        "--raw-files",
        raw_files,
        "--inspect-only",
        "--slice-us",
        str(int(slice_us)),
        "--replay-timing",
        replay_timing,
        "--stats-jsonl",
        str(stats_jsonl),
    ]
    if allow_missing_ext_trigger:
        cmd.append("--allow-missing-ext-trigger")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    parsed = _parse_json_line(proc.stdout)
    return parsed, proc.returncode, proc.stdout, proc.stderr


def _summarize(
    broker_result: Dict[str, Any],
    *,
    expected_period_us: float,
    period_tolerance_us: float,
    replay_sync_anchor_rows: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    cameras = broker_result.get("cameras") or []
    anchor_rows = replay_sync_anchor_rows or {}
    out_cams: Dict[str, Any] = {}
    verified = True
    for cam in cameras:
        alias = str(cam.get("alias") or "")
        count = int(cam.get("triggers_seen") or 0)
        median = cam.get("median_ext_trigger_period_us")
        max_gap = cam.get("max_ext_trigger_gap_us")
        trigger_tracks = cam.get("ext_trigger_tracks") or []
        has_trigger = count > 0
        selected_track: Optional[Dict[str, Any]] = None
        period_ok = False
        for track in trigger_tracks:
            track_median = track.get("median_period_us")
            if not isinstance(track_median, (int, float)) or track_median <= 0:
                continue
            track_period_ok = abs(float(track_median) - float(expected_period_us)) <= float(period_tolerance_us)
            track["period_ok"] = track_period_ok
            if track_period_ok and (
                selected_track is None
                or int(track.get("count") or 0) > int(selected_track.get("count") or 0)
            ):
                selected_track = track
        if selected_track is not None:
            period_ok = True
        elif isinstance(median, (int, float)) and median > 0:
            # Legacy fallback for RAW files where the decoder exposes only one trigger edge stream.
            period_ok = abs(float(median) - float(expected_period_us)) <= float(period_tolerance_us)
        anchor_row_count = int(anchor_rows.get(alias, 0) or 0)
        replay_sync_anchor_ok = bool(anchor_row_count > 1)
        valid = bool(has_trigger and (period_ok or replay_sync_anchor_ok))
        verified = verified and valid
        if period_ok:
            period_check_mode = "trigger_id_polarity_group" if trigger_tracks else "all_trigger_edges"
        elif replay_sync_anchor_ok:
            period_check_mode = "event_trigger_map_anchor_fallback"
        else:
            period_check_mode = "trigger_id_polarity_group" if trigger_tracks else "all_trigger_edges"
        out_cams[alias] = {
            "raw_file": cam.get("raw_file", ""),
            "has_ext_trigger": has_trigger,
            "trigger_count": count,
            "first_trigger_ts_us": cam.get("first_ext_trigger_ts_us", -1),
            "last_trigger_ts_us": cam.get("last_ext_trigger_ts_us", -1),
            "median_period_us": median,
            "max_gap_us": max_gap,
            "period_ok": period_ok,
            "period_check_mode": period_check_mode,
            "selected_trigger_track": selected_track or {},
            "trigger_tracks": trigger_tracks,
            "replay_sync_anchor_ok": replay_sync_anchor_ok,
            "replay_sync_anchor_rows": anchor_row_count,
            "valid_for_replay_sync": valid,
            "last_error": cam.get("last_error", ""),
        }
    return {
        "schema": "event_raw_ext_trigger_check.v1",
        "event_raw_expected_ext_trigger": True,
        "event_raw_ext_trigger_verified": verified,
        "expected_period_us": expected_period_us,
        "period_tolerance_us": period_tolerance_us,
        "event_raw_ext_trigger_count": {k: v["trigger_count"] for k, v in out_cams.items()},
        "event_trigger_map_anchor_rows": dict(sorted(anchor_rows.items())),
        "cameras": out_cams,
        "broker": {
            "returncode": broker_result.get("_returncode"),
            "source_mode": broker_result.get("source_mode", "file_replay"),
            "sync_policy": broker_result.get("sync_policy", "raw_external_trigger_first"),
        },
    }


def _update_manifest(dataset_dir: Path, summary: Dict[str, Any]) -> Path:
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        data = {"dataset_schema": "full_replay_dataset_v1"}
    data["event_raw_expected_ext_trigger"] = summary["event_raw_expected_ext_trigger"]
    data["event_raw_ext_trigger_verified"] = summary["event_raw_ext_trigger_verified"]
    data["event_raw_ext_trigger_count"] = summary["event_raw_ext_trigger_count"]
    data["event_raw_ext_trigger_check"] = summary
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="RAW file(s) or dataset directory/directories.")
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--broker-bin", default="")
    ap.add_argument("--slice-us", type=int, default=4000)
    ap.add_argument("--replay-timing", choices=["realtime", "fast"], default="fast")
    ap.add_argument("--expected-period-us", type=float, default=25000.0)
    ap.add_argument("--period-tolerance-us", type=float, default=2500.0)
    ap.add_argument("--allow-missing-ext-trigger", action="store_true")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--update-manifest", action="store_true")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    broker = _find_broker(project_root, args.broker_bin)
    if broker is None:
        raise SystemExit(
            "event_file_fifo_broker not found. Build native_event_bridge first or pass --broker-bin."
        )

    files = _discover_raw_files(args.paths)
    if not files:
        raise SystemExit("no .raw files found")
    replay_sync_anchor_rows = _discover_replay_sync_anchor_rows(args.paths)

    with tempfile.TemporaryDirectory(prefix="event_raw_ext_trigger_") as td:
        stats_jsonl = Path(td) / "event_file_fifo_broker_stats.jsonl"
        result, rc, stdout, stderr = _run_broker_inspect(
            broker,
            files,
            stats_jsonl=stats_jsonl,
            slice_us=args.slice_us,
            replay_timing=args.replay_timing,
            allow_missing_ext_trigger=args.allow_missing_ext_trigger,
        )
    result["_returncode"] = rc
    summary = _summarize(
        result,
        expected_period_us=args.expected_period_us,
        period_tolerance_us=args.period_tolerance_us,
        replay_sync_anchor_rows=replay_sync_anchor_rows,
    )
    summary["broker"]["stderr_tail"] = "\n".join(stderr.splitlines()[-20:])

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).expanduser().write_text(text + "\n", encoding="utf-8")

    if args.update_manifest:
        first_dir = Path(args.paths[0]).expanduser()
        dataset_dir = first_dir if first_dir.is_dir() else first_dir.parent
        manifest = _update_manifest(dataset_dir.resolve(), summary)
        print(f"[check_event_raw_ext_trigger] manifest updated: {manifest}", file=sys.stderr)

    if not args.allow_missing_ext_trigger and not summary["event_raw_ext_trigger_verified"]:
        return 2
    if rc not in (0,):
        return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
