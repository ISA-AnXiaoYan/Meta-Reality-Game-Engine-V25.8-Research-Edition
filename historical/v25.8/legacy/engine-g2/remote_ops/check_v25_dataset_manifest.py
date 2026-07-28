#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CAMERAS = tuple("ABCDEFGH")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        return {}


def _ordered_existing_dirs(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in paths:
        try:
            path = raw.expanduser().resolve()
        except Exception:
            path = raw
        key = str(path)
        if key in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(key)
        out.append(path)
    return out


def companion_dataset_dirs(dataset_dir: Path) -> List[Path]:
    """Return dataset roots that belong to the same recording session.

    Older V25 data may be split across:
      recordings/<session>  -> MVS AVI/CSV
      raw_records/<session> -> Event RAW
    The operator should still be able to select either folder and get one replay bundle.
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    candidates: List[Path] = [dataset_dir]
    parent = dataset_dir.parent
    session = dataset_dir.name
    if parent.name in {"recordings", "raw_records"}:
        project = parent.parent
        candidates.extend([project / "recordings" / session, project / "raw_records" / session])
    candidates.extend([
        dataset_dir / "recordings" / session,
        dataset_dir / "raw_records" / session,
        dataset_dir / "recordings",
        dataset_dir / "raw_records",
    ])
    return _ordered_existing_dirs(candidates)


def find_manifests(search_roots: Iterable[Path]) -> List[Tuple[Path, Dict[str, Any], str]]:
    out: List[Tuple[Path, Dict[str, Any], str]] = []
    seen: set[str] = set()
    names = [
        ("manifest", "manifest.json"),
        ("recording_manifest", "recording_manifest.json"),
        ("replay_manifest", "replay_manifest.json"),
    ]
    for root in search_roots:
        for kind, name in names:
            path = root / name
            key = str(path.resolve())
            if key in seen or not path.exists():
                continue
            seen.add(key)
            out.append((path, read_json(path), kind))
    return out


def find_manifest(dataset_dir: Path, search_roots: Optional[Iterable[Path]] = None) -> Tuple[Path, Dict[str, Any], str]:
    roots = list(search_roots) if search_roots is not None else [dataset_dir]
    manifests = find_manifests(roots)
    if manifests:
        return manifests[0]
    return dataset_dir / "manifest.json", {}, "missing"


def infer_camera_alias(path: Path, used: set[str]) -> str:
    text = str(path).upper()
    patterns = [
        r"(?:^|[\\/_.-])EVENT[_-]?([A-H])(?:[_-]|[\\/.]|$)",
        r"(?:^|[\\/_.-])MVS[_-]?([A-H])(?:[_-]|[\\/.]|$)",
        r"(?:^|[\\/_.-])CAMERA[_-]?([A-H])(?:[_-]|[\\/.]|$)",
        r"(?:^|[\\/_.-])([A-H])(?:[_-]|[\\/.]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            cam = match.group(1)
            if cam not in used:
                return cam
    for cam in CAMERAS:
        if cam not in used:
            return cam
    return f"cam{len(used)}"


def discover_files(dataset_dir: Path | Iterable[Path], suffixes: Iterable[str], include_tokens: Iterable[str] = ()) -> List[Path]:
    suffix_set = {s.lower() for s in suffixes}
    tokens = [t.lower() for t in include_tokens if t]
    out: List[Path] = []
    roots = [dataset_dir] if isinstance(dataset_dir, Path) else list(dataset_dir)
    seen: set[str] = set()
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffix_set:
                continue
            low = str(path).lower()
            if tokens and not any(token in low for token in tokens):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return sorted(out)


def map_by_camera(files: List[Path]) -> Dict[str, str]:
    used: set[str] = set()
    out: Dict[str, str] = {}
    for path in files:
        cam = infer_camera_alias(path, used)
        used.add(cam)
        out[cam] = str(path.resolve())
    return dict(sorted(out.items()))


def map_by_camera_independent(files: List[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in files:
        cam = infer_camera_alias(path, set(out))
        if cam in CAMERAS and cam not in out:
            out[cam] = str(path.resolve())
    return dict(sorted(out.items()))


def nested_get(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _manifest_path_map(manifest: Dict[str, Any], key: str) -> Dict[str, str]:
    paths = manifest.get("paths", {}) if isinstance(manifest.get("paths"), dict) else {}
    raw = paths.get(key)
    if not isinstance(raw, dict):
        raw = manifest.get(key)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for cam, value in raw.items():
        cam_s = str(cam).upper()
        if cam_s not in CAMERAS:
            continue
        if isinstance(value, dict):
            path = value.get("path") or value.get("file") or value.get("raw_file")
        else:
            path = value
        if path:
            out[cam_s] = str(path)
    return dict(sorted(out.items()))


def _resolve_manifest_paths(dataset_dir: Path, mapping: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cam, raw in mapping.items():
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        out[cam] = str(path.resolve())
    return dict(sorted(out.items()))


def _merged_manifest_path_map(manifests: List[Tuple[Path, Dict[str, Any], str]], key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for manifest_path, manifest, _kind in manifests:
        resolved = _resolve_manifest_paths(manifest_path.parent, _manifest_path_map(manifest, key))
        for cam, path in resolved.items():
            out.setdefault(cam, path)
    return dict(sorted(out.items()))


def _source_root_for_paths(search_roots: List[Path], path_map: Dict[str, str]) -> str:
    if not path_map:
        return ""
    best_root = ""
    best_count = 0
    paths = [Path(raw).expanduser().resolve() for raw in path_map.values()]
    for root in search_roots:
        count = 0
        for path in paths:
            try:
                path.relative_to(root)
                count += 1
            except ValueError:
                pass
        if count > best_count:
            best_count = count
            best_root = str(root.resolve())
    return best_root


def _count_csv_data_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            line_count = sum(1 for _ in fp)
        return max(0, int(line_count) - 1)
    except Exception:
        return -1


def _count_text_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            return sum(1 for line in fp if line.strip())
    except Exception:
        return -1


def _video_info(path: Path) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        opened = bool(cap.isOpened())
        info = {
            "opened": opened,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0,
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0,
        }
        cap.release()
        return info
    except Exception as exc:
        return {"opened": False, "error": f"{type(exc).__name__}: {exc}"}


def _normalized_source_modes(manifest: Dict[str, Any], event_map: Dict[str, str], mvs_map: Dict[str, str]) -> Dict[str, Any]:
    source_modes = manifest.get("source_modes", {}) if isinstance(manifest.get("source_modes"), dict) else {}
    return {
        "event_source_mode": source_modes.get("event_source_mode") or ("file_replay" if event_map else "live_hal"),
        "mvs_source_mode": source_modes.get("mvs_source_mode") or ("file_replay" if mvs_map else "live_camera"),
        "replay_timing": source_modes.get("replay_timing") or manifest.get("replay_timing") or "realtime",
    }


def _manifest_dataset_id(dataset_dir: Path, manifest: Dict[str, Any]) -> str:
    for key in ("dataset_id", "session_id", "recording_id", "run_id"):
        value = manifest.get(key)
        if value:
            return str(value)
    return dataset_dir.resolve().name


def _manifest_session_id(dataset_dir: Path, manifest: Dict[str, Any]) -> str:
    for key in ("session_id", "run_id", "dataset_id", "recording_id"):
        value = manifest.get(key)
        if value:
            return str(value)
    return dataset_dir.resolve().name


def _normalized_sync(manifest: Dict[str, Any]) -> Dict[str, Any]:
    sync = manifest.get("sync", {}) if isinstance(manifest.get("sync"), dict) else {}
    ext_valid = (
        sync.get("event_external_trigger_valid")
        if sync.get("event_external_trigger_valid") is not None
        else manifest.get("event_raw_ext_trigger_verified")
    )
    ext_required = (
        sync.get("event_external_trigger_required")
        if sync.get("event_external_trigger_required") is not None
        else manifest.get("event_raw_expected_ext_trigger")
    )
    root = sync.get("root") or manifest.get("sync_root") or ""
    return {
        "root": root,
        "event_external_trigger_required": bool(ext_required),
        "event_external_trigger_valid": bool(ext_valid),
        "event_trigger_zero_to_mvs_sync": (
            sync.get("event_trigger_zero_to_mvs_sync")
            if isinstance(sync.get("event_trigger_zero_to_mvs_sync"), dict)
            else manifest.get("event_trigger_zero_to_mvs_sync", {})
        ),
        "period_check_mode": (
            sync.get("period_check_mode")
            or manifest.get("period_check_mode")
            or nested_get(manifest, "event_raw_ext_trigger_check", "period_check_mode")
            or "trigger_id_polarity_group"
        ),
        "event_raw_ext_trigger_count": manifest.get("event_raw_ext_trigger_count", {}),
    }


def _merge_sync_from_manifests(sync_norm: Dict[str, Any], manifests: List[Tuple[Path, Dict[str, Any], str]]) -> Dict[str, Any]:
    merged = dict(sync_norm)
    ext_valid_seen = False
    ext_required_seen = False
    ext_counts: Dict[str, Any] = {}
    trigger_zero_anchor: Dict[str, Any] = {}
    for _path, manifest, _kind in manifests:
        sync = manifest.get("sync", {}) if isinstance(manifest.get("sync"), dict) else {}
        ext_valid = (
            sync.get("event_external_trigger_valid")
            if sync.get("event_external_trigger_valid") is not None
            else manifest.get("event_raw_ext_trigger_verified")
        )
        ext_required = (
            sync.get("event_external_trigger_required")
            if sync.get("event_external_trigger_required") is not None
            else manifest.get("event_raw_expected_ext_trigger")
        )
        if ext_valid is True:
            ext_valid_seen = True
        if ext_required is True:
            ext_required_seen = True
        root = sync.get("root") or manifest.get("sync_root")
        if root and not merged.get("root"):
            merged["root"] = root
        count = manifest.get("event_raw_ext_trigger_count", {})
        if isinstance(count, dict):
            ext_counts.update(count)
        anchor = (
            sync.get("event_trigger_zero_to_mvs_sync")
            if isinstance(sync.get("event_trigger_zero_to_mvs_sync"), dict)
            else manifest.get("event_trigger_zero_to_mvs_sync", {})
        )
        if isinstance(anchor, dict):
            for cam, value in anchor.items():
                cam_s = str(cam).upper()
                if cam_s in CAMERAS:
                    trigger_zero_anchor.setdefault(cam_s, value)
    if ext_valid_seen:
        merged["event_external_trigger_valid"] = True
    if ext_required_seen:
        merged["event_external_trigger_required"] = True
    if ext_counts:
        merged["event_raw_ext_trigger_count"] = ext_counts
    if trigger_zero_anchor:
        merged["event_trigger_zero_to_mvs_sync"] = dict(sorted(trigger_zero_anchor.items()))
    return merged


def build_report(dataset_dir: Path, mode: str) -> Dict[str, Any]:
    search_roots = companion_dataset_dirs(dataset_dir)
    manifests = find_manifests(search_roots)
    manifest_path, manifest, manifest_kind = find_manifest(dataset_dir, search_roots)
    event_files = discover_files(search_roots, [".raw"], ["event", "event_raw"])
    if not event_files:
        event_files = discover_files(search_roots, [".raw"])
    mvs_video_files = discover_files(search_roots, [".avi", ".mp4", ".mkv"], ["mvs", "raw_mvs", "mvs_raw"])
    mvs_index_files = discover_files(search_roots, [".csv"], ["frame_index", "mvs"])
    human_files = discover_files(search_roots, [".jsonl"], ["human_result", "human"])
    bullet_files = discover_files(search_roots, [".jsonl"], ["bullet", "pseudo_hit", "hit_candidate"])
    trigger_files = discover_files(search_roots, [".json", ".jsonl", ".csv"], ["trigger", "timeline"])
    event_trigger_map_files = discover_files(search_roots, [".jsonl"], ["event_trigger_map"])

    event_map = _merged_manifest_path_map(manifests, "event_raw") or map_by_camera(event_files)
    mvs_video_map = _merged_manifest_path_map(manifests, "mvs_raw") or map_by_camera_independent(mvs_video_files)
    mvs_index_map = _merged_manifest_path_map(manifests, "mvs_frame_index") or map_by_camera_independent(mvs_index_files)
    event_trigger_map = _merged_manifest_path_map(manifests, "event_trigger_map") or map_by_camera_independent(event_trigger_map_files)
    mvs_map = mvs_video_map
    mvs_video_info = {cam: _video_info(Path(path)) for cam, path in mvs_video_map.items()}
    mvs_index_rows = {cam: _count_csv_data_rows(Path(path)) for cam, path in mvs_index_map.items()}
    event_trigger_map_rows = {cam: _count_text_rows(Path(path)) for cam, path in event_trigger_map.items()}
    source_modes = _normalized_source_modes(manifest, event_map, mvs_map)
    sync_norm = _merge_sync_from_manifests(_normalized_sync(manifest), manifests)
    source_roots = {
        "event_raw": _source_root_for_paths(search_roots, event_map),
        "mvs_raw": _source_root_for_paths(search_roots, mvs_video_map),
        "manifest": str(manifest_path.parent.resolve()) if manifest_kind != "missing" else "",
    }

    missing: List[str] = []
    warnings: List[str] = []

    if manifest_kind == "missing":
        warnings.append("manifest.json missing; using file discovery only")
    if len(search_roots) > 1:
        warnings.append("dataset split bundle detected; merged same-session search roots")
    if mode in {"event_replay", "full_replay"} and not event_map:
        missing.append("event_raw")
    if mode in {"mvs_replay", "full_replay"} and not mvs_map:
        missing.append("mvs_raw")
    if mode in {"event_replay", "full_replay"} and event_map and set(event_map) != set(CAMERAS):
        warnings.append(f"event_raw_camera_count={len(event_map)} expected=8")
    if mode in {"mvs_replay", "full_replay"} and mvs_map and set(mvs_map) != set(CAMERAS):
        warnings.append(f"mvs_raw_camera_count={len(mvs_map)} expected=8")
    if mode in {"mvs_replay", "full_replay"} and mvs_map and set(mvs_index_map) != set(CAMERAS):
        warnings.append(f"mvs_frame_index_camera_count={len(mvs_index_map)} expected=8")
    for cam, info in sorted(mvs_video_info.items()):
        if mode in {"mvs_replay", "full_replay"} and int(info.get("frame_count", 0) or 0) <= 1:
            warnings.append(f"mvs_video_short_camera={cam} frame_count={info.get('frame_count')}")
    for cam, rows in sorted(mvs_index_rows.items()):
        if mode in {"mvs_replay", "full_replay"} and int(rows) <= 1:
            warnings.append(f"mvs_frame_index_short_camera={cam} rows={rows}")
    for cam, path in sorted(event_map.items()):
        if mode in {"event_replay", "full_replay"} and not Path(path).exists():
            warnings.append(f"event_raw_missing_camera={cam}")
    for cam, path in sorted(mvs_video_map.items()):
        if mode in {"mvs_replay", "full_replay"} and not Path(path).exists():
            warnings.append(f"mvs_raw_missing_camera={cam}")
    for cam, path in sorted(mvs_index_map.items()):
        if mode in {"mvs_replay", "full_replay"} and not Path(path).exists():
            warnings.append(f"mvs_frame_index_missing_camera={cam}")
    for cam, path in sorted(event_trigger_map.items()):
        if mode == "full_replay" and not Path(path).exists():
            warnings.append(f"event_trigger_map_missing_camera={cam}")

    if mode in {"event_replay", "full_replay"}:
        if not sync_norm.get("root"):
            warnings.append("sync.root missing; assuming mvs_soft_trigger for current V25 event replay contract")
            sync_norm["root"] = "mvs_soft_trigger"
        elif sync_norm.get("root") != "mvs_soft_trigger":
            warnings.append(f"sync.root={sync_norm.get('root')!r} expected='mvs_soft_trigger'")
        if sync_norm.get("event_external_trigger_required") is False:
            warnings.append("event_external_trigger_required=false")
        if sync_norm.get("event_external_trigger_valid") is False:
            warnings.append("event_external_trigger_valid=false")

    trigger_zero_anchor = sync_norm.get("event_trigger_zero_to_mvs_sync", {})
    if not isinstance(trigger_zero_anchor, dict):
        trigger_zero_anchor = {}
    trigger_zero_anchor_ok = set(str(k).upper() for k in trigger_zero_anchor.keys()) == set(CAMERAS)
    trigger_map_paths_ok = bool(
        set(event_trigger_map) == set(CAMERAS)
        and all(Path(path).exists() for path in event_trigger_map.values())
        and all(int(rows) > 1 for rows in event_trigger_map_rows.values())
    )
    event_mvs_sync_anchor_ok = bool(trigger_zero_anchor_ok or trigger_map_paths_ok)
    event_mvs_sync_anchor_source = (
        "event_trigger_zero_to_mvs_sync"
        if trigger_zero_anchor_ok
        else ("event_trigger_map_jsonl" if trigger_map_paths_ok else "")
    )
    if mode == "full_replay" and not event_mvs_sync_anchor_ok:
        warnings.append("full_replay_sync_anchor_missing")

    ok = not missing
    validation = manifest.get("validation", {}) if isinstance(manifest.get("validation"), dict) else {}
    event_paths_ok = all(Path(path).exists() for path in event_map.values())
    usable_event = bool(
        set(event_map) == set(CAMERAS)
        and event_paths_ok
        and sync_norm.get("event_external_trigger_valid") is True
    )
    mvs_video_lengths_ok = all(int(info.get("frame_count", 0) or 0) > 1 for info in mvs_video_info.values())
    mvs_index_lengths_ok = all(int(rows) > 1 for rows in mvs_index_rows.values())
    usable_mvs = bool(
        set(mvs_video_map) == set(CAMERAS)
        and set(mvs_index_map) == set(CAMERAS)
        and mvs_video_lengths_ok
        and mvs_index_lengths_ok
    )
    degraded_reasons = list(validation.get("degraded_reasons", []))
    if set(mvs_video_map) == set(CAMERAS) and not mvs_video_lengths_ok:
        degraded_reasons.append("mvs_video_short_camera")
    if set(mvs_index_map) == set(CAMERAS) and not mvs_index_lengths_ok:
        degraded_reasons.append("mvs_frame_index_short_camera")
    if usable_event and usable_mvs and not event_mvs_sync_anchor_ok:
        degraded_reasons.append("full_replay_sync_anchor_missing")
    validation_norm = {
        "usable_for_event_file_replay": bool(usable_event),
        "usable_for_mvs_file_replay": bool(usable_mvs),
        "usable_for_full_replay": bool(usable_event and usable_mvs),
        "usable_for_full_replay_startup": bool(usable_event and usable_mvs),
        "usable_for_full_replay_sync": bool(usable_event and usable_mvs and event_mvs_sync_anchor_ok),
        "event_mvs_sync_anchor_ok": bool(event_mvs_sync_anchor_ok),
        "event_mvs_sync_anchor_source": event_mvs_sync_anchor_source,
        "degraded_reasons": sorted(set(str(x) for x in degraded_reasons if str(x))),
    }
    return {
        "schema_version": 1,
        "kind": "v25_dataset_manifest_check",
        "mode": mode,
        "ok": ok,
        "dataset_id": _manifest_dataset_id(dataset_dir, manifest),
        "session_id": _manifest_session_id(dataset_dir, manifest),
        "dataset_dir": str(dataset_dir.resolve()),
        "search_roots": [str(p.resolve()) for p in search_roots],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_kind": manifest_kind,
        "manifest_sources": [
            {"kind": kind, "path": str(path.resolve())}
            for path, _manifest_obj, kind in manifests
        ],
        "missing": missing,
        "warnings": warnings,
        "source_modes": source_modes,
        "source_roots": source_roots,
        "sync": sync_norm,
        "validation": validation_norm,
        "paths": {
            "event_raw_by_camera": event_map,
            "mvs_raw_by_camera": mvs_map,
            "mvs_video_by_camera": mvs_video_map,
            "mvs_frame_index_by_camera": mvs_index_map,
            "mvs_video_info_by_camera": mvs_video_info,
            "mvs_frame_index_rows_by_camera": mvs_index_rows,
            "event_trigger_map_by_camera": event_trigger_map,
            "event_trigger_map_rows_by_camera": event_trigger_map_rows,
            "human_polygon_files": [str(p.resolve()) for p in human_files[:64]],
            "event_track_files": [str(p.resolve()) for p in bullet_files[:64]],
            "trigger_timeline_files": [str(p.resolve()) for p in trigger_files[:64]],
        },
        "counts": {
            "event_raw": len(event_files),
            "mvs_raw": len(mvs_video_files),
            "mvs_frame_index": len(mvs_index_files),
            "event_trigger_map": len(event_trigger_map_files),
            "human_polygon": len(human_files),
            "event_tracks": len(bullet_files),
            "trigger_timeline": len(trigger_files),
        },
    }


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _scenario_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "dataset_label": str(getattr(args, "dataset_label", "") or ""),
        "scenario_name": str(getattr(args, "scenario_name", "") or ""),
        "people_count": _int_or_none(getattr(args, "people_count", None)),
        "shooting": str(getattr(args, "shooting", "unknown") or "unknown"),
        "hit_events": _int_or_none(getattr(args, "hit_events", None)),
        "miss_events": _int_or_none(getattr(args, "miss_events", None)),
        "manual_label_file": str(getattr(args, "manual_label_file", "") or ""),
        "operator_notes": str(getattr(args, "operator_notes", "") or ""),
    }


def _compact_path_map(paths: Dict[str, Any], key: str) -> Dict[str, str]:
    raw = paths.get(key) if isinstance(paths, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in sorted(raw.items()) if v}


def build_resolved_manifest(
    report: Dict[str, Any],
    *,
    replay_timing: str = "",
    scenario: Optional[Dict[str, Any]] = None,
    allow_missing_ext_trigger: bool = False,
    event_loop: bool = True,
    mvs_loop: bool = True,
) -> Dict[str, Any]:
    """Build the stable replay contract that launchers, status UIs, and reports share.

    The checker report is intentionally verbose and diagnostic.  This resolved manifest is
    the operator-facing contract: what data exists, what modes it can support, and the
    exact startup arguments needed to reproduce the bundle.
    """
    paths = report.get("paths", {}) if isinstance(report.get("paths"), dict) else {}
    validation = report.get("validation", {}) if isinstance(report.get("validation"), dict) else {}
    timing = str(replay_timing or nested_get(report, "source_modes", "replay_timing") or "realtime")
    capabilities = {
        "event_file_replay": bool(validation.get("usable_for_event_file_replay")),
        "mvs_file_replay": bool(validation.get("usable_for_mvs_file_replay")),
        "full_replay": bool(validation.get("usable_for_full_replay")),
        "full_replay_startup": bool(validation.get("usable_for_full_replay_startup")),
        "full_replay_sync": bool(validation.get("usable_for_full_replay_sync")),
        "event_mvs_sync_anchor_ok": bool(validation.get("event_mvs_sync_anchor_ok")),
        "event_mvs_sync_anchor_source": str(validation.get("event_mvs_sync_anchor_source") or ""),
    }
    scenario_norm = dict(scenario or {})
    for key, default in {
        "dataset_label": "",
        "scenario_name": "",
        "people_count": None,
        "shooting": "unknown",
        "hit_events": None,
        "miss_events": None,
        "manual_label_file": "",
        "operator_notes": "",
    }.items():
        scenario_norm.setdefault(key, default)
    return {
        "schema_version": 1,
        "kind": "v25_dataset_resolved_manifest",
        "resolved_at_wall_us": int(time.time() * 1_000_000),
        "dataset_id": str(report.get("dataset_id") or Path(str(report.get("dataset_dir") or ".")).name),
        "session_id": str(report.get("session_id") or Path(str(report.get("dataset_dir") or ".")).name),
        "dataset_dir": str(report.get("dataset_dir") or ""),
        "search_roots": list(report.get("search_roots", [])) if isinstance(report.get("search_roots"), list) else [],
        "scenario": scenario_norm,
        "source_modes": report.get("source_modes", {}) if isinstance(report.get("source_modes"), dict) else {},
        "source_roots": report.get("source_roots", {}) if isinstance(report.get("source_roots"), dict) else {},
        "sync": report.get("sync", {}) if isinstance(report.get("sync"), dict) else {},
        "capabilities": capabilities,
        "validation": validation,
        "paths": {
            "event_raw_by_camera": _compact_path_map(paths, "event_raw_by_camera"),
            "mvs_raw_by_camera": _compact_path_map(paths, "mvs_raw_by_camera"),
            "mvs_frame_index_by_camera": _compact_path_map(paths, "mvs_frame_index_by_camera"),
            "event_trigger_map_by_camera": _compact_path_map(paths, "event_trigger_map_by_camera"),
            "human_polygon_files": list(paths.get("human_polygon_files", [])) if isinstance(paths.get("human_polygon_files"), list) else [],
            "event_track_files": list(paths.get("event_track_files", [])) if isinstance(paths.get("event_track_files"), list) else [],
            "trigger_timeline_files": list(paths.get("trigger_timeline_files", [])) if isinstance(paths.get("trigger_timeline_files"), list) else [],
        },
        "counts": report.get("counts", {}) if isinstance(report.get("counts"), dict) else {},
        "manifest_sources": report.get("manifest_sources", []) if isinstance(report.get("manifest_sources"), list) else [],
        "warnings": list(report.get("warnings", [])) if isinstance(report.get("warnings"), list) else [],
        "missing": list(report.get("missing", [])) if isinstance(report.get("missing"), list) else [],
        "degraded_reasons": list(validation.get("degraded_reasons", [])) if isinstance(validation.get("degraded_reasons"), list) else [],
        "recommended_launch": {
            "replay_timing": timing,
            "event_replay_args": launch_extra_args(
                report,
                "event_replay",
                replay_timing=timing,
                event_loop=event_loop,
                allow_missing_ext_trigger=allow_missing_ext_trigger,
            ),
            "mvs_replay_args": launch_extra_args(
                report,
                "mvs_replay",
                replay_timing=timing,
                mvs_loop=mvs_loop,
            ),
            "full_replay_args": launch_extra_args(
                report,
                "full_replay",
                replay_timing=timing,
                event_loop=event_loop,
                mvs_loop=mvs_loop,
                allow_missing_ext_trigger=allow_missing_ext_trigger,
            ),
        },
    }


def write_resolved_manifest(path_arg: str, dataset_dir: Path, resolved: Dict[str, Any]) -> Path:
    out = dataset_dir / "replay_manifest_resolved.json" if path_arg == "auto" else Path(path_arg).expanduser()
    if not out.is_absolute():
        out = dataset_dir / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out.resolve()


def event_raw_mapping(report: Dict[str, Any]) -> str:
    mapping = report.get("paths", {}).get("event_raw_by_camera", {})
    if not isinstance(mapping, dict):
        return ""
    return ",".join(f"{cam}={path}" for cam, path in sorted(mapping.items()))


def _usable_key_for_mode(mode: str) -> str:
    if mode == "event_replay":
        return "usable_for_event_file_replay"
    if mode == "mvs_replay":
        return "usable_for_mvs_file_replay"
    if mode == "full_replay":
        return "usable_for_full_replay"
    return ""


def is_usable_for_mode(report: Dict[str, Any], mode: str) -> bool:
    key = _usable_key_for_mode(mode)
    if not key:
        return True
    validation = report.get("validation", {}) if isinstance(report.get("validation"), dict) else {}
    return bool(validation.get(key))


def launch_extra_argv(
    report: Dict[str, Any],
    mode: str,
    replay_timing: str = "",
    event_loop: bool = True,
    mvs_loop: bool = True,
    allow_missing_ext_trigger: bool = False,
) -> List[str]:
    source_modes = report.get("source_modes", {}) if isinstance(report.get("source_modes"), dict) else {}
    source_roots = report.get("source_roots", {}) if isinstance(report.get("source_roots"), dict) else {}
    dataset_dir = str(report.get("dataset_dir") or "")
    mvs_dataset_dir = str(source_roots.get("mvs_raw") or dataset_dir)
    timing = str(replay_timing or source_modes.get("replay_timing") or "realtime")
    parts: List[str] = []
    if dataset_dir:
        parts += ["--dataset-dir", dataset_dir]
    if mode in {"event_replay", "full_replay"}:
        mapping = event_raw_mapping(report)
        if not mapping:
            return []
        parts += [
            "--event-source-mode", "file_replay",
            "--event-file-fifo-broker-raw-files", mapping,
            "--event-file-fifo-broker-replay-timing", timing,
        ]
        if allow_missing_ext_trigger:
            parts += ["--event-file-fifo-broker-allow-missing-ext-trigger"]
        if event_loop:
            parts += ["--event-file-fifo-broker-loop"]
    if mode in {"mvs_replay", "full_replay"}:
        if not mvs_dataset_dir:
            return []
        parts += [
            "--mvs-source-mode", "file_replay",
            "--mvs-file-frame-broker-dataset-dir", mvs_dataset_dir,
            "--mvs-file-frame-broker-replay-timing", timing,
        ]
        if mvs_loop:
            parts += ["--mvs-file-frame-broker-loop"]
    return parts


def launch_extra_args(
    report: Dict[str, Any],
    mode: str,
    replay_timing: str = "",
    event_loop: bool = True,
    mvs_loop: bool = True,
    allow_missing_ext_trigger: bool = False,
) -> str:
    """Return the legacy whitespace-delimited launch contract."""
    return " ".join(launch_extra_argv(
        report,
        mode,
        replay_timing=replay_timing,
        event_loop=event_loop,
        mvs_loop=mvs_loop,
        allow_missing_ext_trigger=allow_missing_ext_trigger,
    ))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check a V25 dataset manifest and infer replay inputs.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--mode", choices=["event_replay", "mvs_replay", "full_replay", "live"], default="full_replay")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--print-resolved-manifest", action="store_true")
    ap.add_argument("--write-resolved-manifest", nargs="?", const="auto", default="")
    ap.add_argument("--print-event-raw-mapping", action="store_true")
    ap.add_argument("--print-launch-extra-args", action="store_true")
    ap.add_argument("--require-usable", action="store_true")
    ap.add_argument("--replay-timing", choices=["realtime", "fast", ""], default="")
    ap.add_argument("--mvs-loop", action="store_true", default=True)
    ap.add_argument("--no-mvs-loop", dest="mvs_loop", action="store_false")
    ap.add_argument("--allow-missing-ext-trigger", action="store_true", default=False)
    ap.add_argument("--dataset-label", default="")
    ap.add_argument("--scenario-name", default="")
    ap.add_argument("--people-count", type=int, default=None)
    ap.add_argument("--shooting", choices=["unknown", "none", "present"], default="unknown")
    ap.add_argument("--hit-events", type=int, default=None)
    ap.add_argument("--miss-events", type=int, default=None)
    ap.add_argument("--manual-label-file", default="")
    ap.add_argument("--operator-notes", default="")
    args = ap.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    report = build_report(dataset_dir, args.mode)
    resolved = build_resolved_manifest(
        report,
        replay_timing=args.replay_timing,
        scenario=_scenario_from_args(args),
        allow_missing_ext_trigger=bool(args.allow_missing_ext_trigger),
        event_loop=True,
        mvs_loop=bool(args.mvs_loop),
    )
    if args.write_resolved_manifest:
        write_resolved_manifest(str(args.write_resolved_manifest), dataset_dir, resolved)

    if args.print_event_raw_mapping:
        print(event_raw_mapping(report))
        return 0 if report["paths"]["event_raw_by_camera"] else 2

    if args.require_usable and not is_usable_for_mode(report, args.mode):
        validation = report.get("validation", {}) if isinstance(report.get("validation"), dict) else {}
        reasons = validation.get("degraded_reasons", [])
        print(json.dumps({
            "ok": False,
            "mode": args.mode,
            "usable_key": _usable_key_for_mode(args.mode),
            "validation": validation,
            "missing": report.get("missing", []),
            "warnings": report.get("warnings", []),
            "degraded_reasons": reasons,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3

    if args.print_launch_extra_args:
        extra = launch_extra_args(
            report,
            args.mode,
            replay_timing=args.replay_timing,
            event_loop=True,
            mvs_loop=bool(args.mvs_loop),
            allow_missing_ext_trigger=bool(args.allow_missing_ext_trigger),
        )
        print(extra)
        return 0 if extra else 2

    payload = resolved if args.print_resolved_manifest else report
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).expanduser().write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
