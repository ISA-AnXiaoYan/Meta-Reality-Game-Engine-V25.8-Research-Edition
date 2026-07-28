#!/usr/bin/env python3
"""Analyze BEV event-layer sync truth against event/MVS trigger maps.

This is a diagnostic helper for V24-A.  It is intentionally read-only and can
work on either a live sync_ipc directory or an archived run directory that has
the relevant status files copied into it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_jsonl(path: Path, max_lines: int = 200000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        pass
    return out


def _anchor_ts(rec: Dict[str, Any]) -> int:
    return _safe_int(rec.get("mvs_soft_trigger_event_ts_us", rec.get("event_edge_ts_us")), 0)


def _usable_records(records: Iterable[Dict[str, Any]]) -> List[Tuple[int, float, Dict[str, Any]]]:
    out: List[Tuple[int, float, Dict[str, Any]]] = []
    for rec in records:
        mapped = rec.get("mapped_mvs_sync_index")
        if mapped is None:
            continue
        ts = _anchor_ts(rec)
        if ts <= 0:
            continue
        out.append((int(ts), float(mapped), rec))
    out.sort(key=lambda row: row[0])
    return out


def _map_event_ts(usable: List[Tuple[int, float, Dict[str, Any]]], event_ts_us: int) -> Optional[Tuple[float, str, float]]:
    if not usable or event_ts_us <= 0:
        return None
    before = None
    after = None
    for row in usable:
        if row[0] <= event_ts_us:
            before = row
        if row[0] >= event_ts_us:
            after = row
            break
    nearest = min(usable, key=lambda row: abs(row[0] - event_ts_us))
    nearest_delta_ms = abs(float(nearest[0] - event_ts_us)) / 1000.0
    if before is not None and after is not None:
        if before[0] == after[0]:
            return float(before[1]), "exact", nearest_delta_ms
        frac = (float(event_ts_us) - float(before[0])) / max(1.0, float(after[0] - before[0]))
        mapped = float(before[1]) + frac * (float(after[1]) - float(before[1]))
        return mapped, "interpolated", nearest_delta_ms
    return float(nearest[1]), "nearest", nearest_delta_ms


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return math.nan
    vals = sorted(vals)
    k = (len(vals) - 1) * float(p)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def _camera_paths(root: Path, cam: str) -> Dict[str, Path]:
    status_dir = root if root.name == "status" else root / "status"
    return {
        "ring": root / f"event_sync_map_ring_{cam}.json",
        "ring_status": status_dir / f"event_sync_map_ring_{cam}.json",
        "map": root / f"event_trigger_map_{cam}.jsonl",
        "map_status": status_dir / f"event_trigger_map_{cam}.jsonl",
        "history": root / f"event_overlay_masked_{cam}_history.json",
        "history_status": status_dir / f"event_overlay_masked_{cam}_history.json",
    }


def analyze(root: Path, cams: List[str], tick_ms: float) -> Dict[str, Any]:
    per_cam: Dict[str, Any] = {}
    for cam in cams:
        paths = _camera_paths(root, cam)
        ring_path = paths["ring"] if paths["ring"].exists() else paths["ring_status"]
        map_path = paths["map"] if paths["map"].exists() else paths["map_status"]
        history_path = paths["history"] if paths["history"].exists() else paths["history_status"]

        ring = _load_json(ring_path) if ring_path.exists() else {}
        records = ring.get("records", []) if isinstance(ring.get("records"), list) else []
        source = "ring" if records else "jsonl"
        if not records:
            records = _load_jsonl(map_path)
        usable = _usable_records(records)

        history = _load_json(history_path) if history_path.exists() else {}
        entries = history.get("entries", []) if isinstance(history.get("entries"), list) else []
        deltas_ms: List[float] = []
        nearest_ms: List[float] = []
        mapped_count = 0
        by_mode: Dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("valid", True):
                continue
            event_ts = _safe_int(entry.get("event_window_center_us", entry.get("event_ts_us")), 0)
            mapped = _map_event_ts(usable, event_ts)
            if mapped is None:
                continue
            mapped_f, mode, nearest_delta = mapped
            mapped_count += 1
            by_mode[mode] = by_mode.get(mode, 0) + 1
            target = _safe_float(entry.get("mapped_mvs_sync_index"), math.nan)
            if math.isfinite(target):
                deltas_ms.append((mapped_f - target) * float(tick_ms))
            nearest_ms.append(float(nearest_delta))

        per_cam[cam] = {
            "source": source,
            "ring_path": str(ring_path),
            "map_path": str(map_path),
            "history_path": str(history_path),
            "records": len(records),
            "usable_records": len(usable),
            "history_entries": len(entries),
            "mapped_history_entries": int(mapped_count),
            "modes": by_mode,
            "nearest_anchor_delta_ms_p50": None if not nearest_ms else round(_percentile(nearest_ms, 0.50), 3),
            "nearest_anchor_delta_ms_p95": None if not nearest_ms else round(_percentile(nearest_ms, 0.95), 3),
            "sidecar_mapped_delta_ms_p95": None if not deltas_ms else round(_percentile([abs(v) for v in deltas_ms], 0.95), 3),
        }
    return {"root": str(root), "tick_ms": float(tick_ms), "per_camera": per_cam}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_or_sync_root", type=Path)
    ap.add_argument("--cams", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--tick-ms", type=float, default=25.0)
    args = ap.parse_args()
    cams = [c.strip() for c in str(args.cams).split(",") if c.strip()]
    result = analyze(args.run_or_sync_root, cams, args.tick_ms)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
