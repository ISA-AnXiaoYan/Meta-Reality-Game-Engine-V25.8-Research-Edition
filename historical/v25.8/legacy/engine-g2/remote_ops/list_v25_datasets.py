#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RAW_SUFFIXES = {".raw"}
MVS_SUFFIXES = {".avi", ".mp4", ".mkv"}
JSONL_SUFFIXES = {".jsonl"}
MANIFEST_NAMES = {"manifest.json", "recording_manifest.json", "replay_manifest.json", "summary.json"}


def _human_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}PB"


def _safe_stat(path: Path) -> Optional[Any]:
    try:
        return path.stat()
    except Exception:
        return None


def _iter_files_limited(root: Path, max_files: int = 5000) -> Iterable[Path]:
    count = 0
    try:
        iterator = root.rglob("*")
    except Exception:
        return
    for path in iterator:
        if count >= max_files:
            return
        try:
            if path.is_file():
                count += 1
                yield path
        except Exception:
            continue


def _read_manifest(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return {}


def _first_manifest(root: Path) -> Optional[Path]:
    for name in MANIFEST_NAMES:
        path = root / name
        if path.exists():
            return path
    for path in root.glob("manifest*.json"):
        if path.is_file():
            return path
    return None


def summarize_candidate(root: Path, project: Path, max_files: int = 5000) -> Dict[str, Any]:
    counts = {
        "event_raw": 0,
        "mvs_video": 0,
        "human_polygon": 0,
        "event_tracks": 0,
        "hit_debug": 0,
        "manifest": 0,
    }
    examples: Dict[str, List[str]] = {key: [] for key in counts}
    total_size = 0
    newest_mtime = 0.0
    scanned = 0

    manifest_path = _first_manifest(root)
    manifest = _read_manifest(manifest_path) if manifest_path else {}
    if manifest_path:
        counts["manifest"] = 1
        examples["manifest"].append(str(manifest_path.relative_to(project)) if manifest_path.is_relative_to(project) else str(manifest_path))

    for path in _iter_files_limited(root, max_files=max_files):
        scanned += 1
        st = _safe_stat(path)
        if st:
            total_size += int(st.st_size)
            newest_mtime = max(newest_mtime, float(st.st_mtime))
        low = str(path).lower()
        suffix = path.suffix.lower()
        category = ""
        if suffix in RAW_SUFFIXES and ("event" in low or "event_raw" in low):
            category = "event_raw"
        elif suffix in MVS_SUFFIXES and ("mvs" in low or "raw_mvs" in low):
            category = "mvs_video"
        elif suffix in JSONL_SUFFIXES and ("human" in low or "polygon" in low):
            category = "human_polygon"
        elif suffix in JSONL_SUFFIXES and ("bullet" in low or "track" in low):
            category = "event_tracks"
        elif suffix in JSONL_SUFFIXES and ("hit" in low or "pseudo" in low or "reject" in low):
            category = "hit_debug"
        if category:
            counts[category] += 1
            if len(examples[category]) < 4:
                try:
                    examples[category].append(str(path.relative_to(project)))
                except Exception:
                    examples[category].append(str(path))

    if newest_mtime <= 0.0:
        st_root = _safe_stat(root)
        newest_mtime = float(st_root.st_mtime) if st_root else 0.0

    flags = {
        "usable_for_event_replay_guess": counts["event_raw"] >= 1,
        "usable_for_mvs_replay_guess": counts["mvs_video"] >= 1,
        "has_human_polygon": counts["human_polygon"] >= 1,
        "has_event_tracks": counts["event_tracks"] >= 1,
        "has_hit_debug": counts["hit_debug"] >= 1,
    }
    if manifest:
        validation = manifest.get("validation", {}) if isinstance(manifest.get("validation"), dict) else {}
        sync = manifest.get("sync", {}) if isinstance(manifest.get("sync"), dict) else {}
        flags["manifest_event_ext_trigger_valid"] = bool(
            sync.get("event_external_trigger_valid", manifest.get("event_raw_ext_trigger_verified", False))
        )
        for key in ("usable_for_event_file_replay", "usable_for_mvs_file_replay", "usable_for_full_replay"):
            if key in validation:
                flags[key] = bool(validation.get(key))

    rel = str(root.relative_to(project)) if root.is_relative_to(project) else str(root)
    return {
        "path": str(root),
        "relative_path": rel,
        "name": root.name,
        "mtime": newest_mtime,
        "age_hours": round(max(0.0, time.time() - newest_mtime) / 3600.0, 3) if newest_mtime else None,
        "total_size": total_size,
        "total_size_human": _human_size(total_size),
        "scanned_files": scanned,
        "counts": counts,
        "examples": examples,
        "flags": flags,
        "manifest_kind": manifest.get("kind", "") if isinstance(manifest, dict) else "",
        "dataset_kind": manifest.get("dataset_kind", manifest.get("scenario", "")) if isinstance(manifest, dict) else "",
        "session_id": manifest.get("session_id", "") if isinstance(manifest, dict) else "",
    }


def candidate_dirs(project: Path) -> List[Path]:
    roots = [
        project / "raw_records",
        project / "recordings",
        project / "recordings" / "person_id_datasets",
        project / "recordings" / "evidence_replay_datasets",
        project / "sync_ipc" / "full_system_tests",
    ]
    out: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        if root not in seen:
            out.append(root)
            seen.add(root)
        try:
            children = [p for p in root.iterdir() if p.is_dir()]
        except Exception:
            children = []
        children.sort(key=lambda p: _safe_stat(p).st_mtime if _safe_stat(p) else 0.0, reverse=True)
        for child in children[:80]:
            if child not in seen:
                out.append(child)
                seen.add(child)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="List likely V25 replay/recording dataset folders.")
    ap.add_argument("--project", default=".")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-files-per-candidate", type=int, default=5000)
    args = ap.parse_args(argv)

    project = Path(args.project).expanduser().resolve()
    rows = []
    for root in candidate_dirs(project):
        row = summarize_candidate(root, project=project, max_files=max(100, args.max_files_per_candidate))
        counts = row.get("counts", {})
        if (
                counts.get("event_raw", 0)
                or counts.get("mvs_video", 0)
                or counts.get("human_polygon", 0)
                or counts.get("event_tracks", 0)
                or counts.get("hit_debug", 0)
                or counts.get("manifest", 0)):
            rows.append(row)
    rows.sort(key=lambda x: float(x.get("mtime", 0.0) or 0.0), reverse=True)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(json.dumps({
        "kind": "v25_dataset_candidates",
        "project": str(project),
        "limit": args.limit,
        "count": len(rows),
        "candidates": rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
