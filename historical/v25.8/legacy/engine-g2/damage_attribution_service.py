#!/usr/bin/env python3
"""Attribute confirmed hit candidates to runtime global IDs.

This sidecar does not map players, names, teams, or watches. It only converts
local hit candidates and global-person tracks into runtime global IDs:
source_global_id -> target_global_id.

When source attribution is not enabled yet, direct_hit_global_id mode publishes
the optimized final hit candidate with a fixed source global ID while still
binding the victim to the current Global Person ID service.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_gid_int(value: Any, default: int = -1) -> int:
    if isinstance(value, str):
        raw = value.strip().upper()
        if raw.startswith("G"):
            raw = raw[1:]
        return _safe_int(raw, default)
    return _safe_int(value, default)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _gid_str(gid: int) -> str:
    return f"G{int(gid):04d}"


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _resolve_path(sync_root: Path, project_root: Path, value: str) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    if str(value).startswith("sync_ipc/") or str(value).startswith("sync_ipc\\"):
        return project_root / p
    return sync_root / p


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _xy(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, dict):
        x = _safe_float(value.get("x"))
        y = _safe_float(value.get("y"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _safe_float(value[0])
        y = _safe_float(value[1])
    else:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return float(x), float(y)


class JsonlTailer:
    def __init__(self, path: Path, read_existing: bool = False) -> None:
        self.path = path
        self.offset = 0
        self.error = ""
        self._initialized = False
        self.read_existing = read_existing

    def read_new(self, max_lines: int = 1000) -> List[Dict[str, Any]]:
        if not self.path.exists():
            self.error = "missing"
            return []
        try:
            size = self.path.stat().st_size
            if not self._initialized:
                self.offset = 0 if self.read_existing else size
                self._initialized = True
            if size < self.offset:
                self.offset = 0
            out: List[Dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8", errors="replace") as fp:
                fp.seek(self.offset)
                for _ in range(max_lines):
                    line = fp.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        pass
                self.offset = fp.tell()
            self.error = ""
            return out
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []


class DamageAttributionService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.sync_root = Path(args.sync_root).resolve()
        self.hit_path = _resolve_path(self.sync_root, self.project_root, args.hit_jsonl)
        self.track_latest_path = _resolve_path(self.sync_root, self.project_root, args.track_latest_json)
        self.person_latest_path = _resolve_path(self.sync_root, self.project_root, args.person_latest_json)
        self.global_bev_status_path = _resolve_path(self.sync_root, self.project_root, args.global_bev_status_json)
        self.events_path = _resolve_path(self.sync_root, self.project_root, args.events_jsonl)
        self.latest_path = _resolve_path(self.sync_root, self.project_root, args.latest_json)
        self.status_path = _resolve_path(self.sync_root, self.project_root, args.status_json)
        self.hit_tailer = JsonlTailer(self.hit_path, bool(args.read_existing))
        self.stats = Counter()
        self.event_seen: Dict[str, float] = {}
        self.latest_event: Dict[str, Any] = {}
        self.person_history: deque = deque(maxlen=max(1, int(getattr(args, "person_history_max_snapshots", 2048))))
        self.display_history: deque = deque(maxlen=max(1, int(getattr(args, "display_history_max_snapshots", 2048))))
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_fp = self.events_path.open("a", encoding="utf-8", buffering=1)

    @property
    def mode_enabled(self) -> bool:
        return str(self.args.mode) not in {"disabled", "off"}

    @property
    def direct_hit_global_id_mode(self) -> bool:
        return str(getattr(self.args, "attribution_source", "global_bullet") or "global_bullet") == "direct_hit_global_id"

    def _load_person_tracks(self) -> List[Dict[str, Any]]:
        obj = _read_json(self.person_latest_path) or {}
        tracks = obj.get("tracks", [])
        if not isinstance(tracks, list):
            return []
        now = _now_ms()
        out: List[Dict[str, Any]] = []
        max_age = float(self.args.person_max_age_ms)
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            age = _safe_float(tr.get("age_ms"), 0.0)
            if max_age >= 0 and age > max_age:
                continue
            gid = _safe_int(tr.get("global_id"), -1)
            if gid < 0:
                continue
            xy = _xy(tr.get("field_xy") or tr.get("filtered_xy") or tr.get("raw_xy"))
            if xy is None:
                continue
            item = dict(tr)
            item["_gid"] = gid
            item["_xy"] = xy
            item["_loaded_ms"] = now
            out.append(item)
        sync_index = _safe_int(obj.get("sync_index"), -1)
        timestamp_ms = _safe_int(obj.get("timestamp_ms"), now)
        self._remember_person_snapshot(sync_index, timestamp_ms, out)
        return out

    def _remember_person_snapshot(self, sync_index: int, timestamp_ms: int, persons: List[Dict[str, Any]]) -> None:
        snap = {
            "sync_index": int(sync_index),
            "timestamp_ms": int(timestamp_ms if timestamp_ms >= 0 else _now_ms()),
            "loaded_ms": _now_ms(),
            "tracks": [dict(p) for p in persons],
        }
        if self.person_history:
            last = self.person_history[-1]
            if int(last.get("sync_index", -999999)) == int(snap["sync_index"]):
                last["timestamp_ms"] = snap["timestamp_ms"]
                last["loaded_ms"] = snap["loaded_ms"]
                last["tracks"] = snap["tracks"]
                return
        self.person_history.append(snap)

    def _load_bullet_tracks(self) -> List[Dict[str, Any]]:
        obj = _read_json(self.track_latest_path) or {}
        tracks = obj.get("tracks", [])
        if not isinstance(tracks, list):
            return []
        return [t for t in tracks if isinstance(t, dict)]

    def _candidate_camera(self, cand: Dict[str, Any]) -> str:
        return str(cand.get("camera") or cand.get("camera_alias") or cand.get("pair_id") or "").strip()

    def _candidate_bullet_id(self, cand: Dict[str, Any]) -> int:
        return _safe_int(cand.get("bullet_id"), _safe_int(cand.get("display_bullet_id"), 0))

    def _candidate_local_id(self, cand: Dict[str, Any]) -> int:
        identity = cand.get("target_identity_evidence") if isinstance(cand.get("target_identity_evidence"), dict) else {}
        for key in ("target_camera_local_id", "camera_local_id"):
            val = _safe_int(identity.get(key), -1)
            if val >= 0:
                return val
        return _safe_int(cand.get("target_camera_local_id", cand.get("stable_id", cand.get("person_id"))), -1)

    def _candidate_sync_index(self, cand: Dict[str, Any]) -> int:
        identity = cand.get("target_identity_evidence") if isinstance(cand.get("target_identity_evidence"), dict) else {}
        for key in ("target_person_sync_index", "person_sync_index"):
            val = _safe_int(identity.get(key), -1)
            if val >= 0:
                return val
        for key in ("target_person_sync_index", "person_frame_sync_index", "mvs_sync_index", "impact_sync_index"):
            val = _safe_int(cand.get(key), -1)
            if val >= 0:
                return val
        return -1

    def _candidate_explicit_runtime_gid(self, cand: Dict[str, Any]) -> Tuple[int, str]:
        identity = cand.get("target_identity_evidence") if isinstance(cand.get("target_identity_evidence"), dict) else {}
        for key in ("target_global_runtime_id", "target_global_runtime_id_int", "target_source_global_id_int"):
            gid = _safe_gid_int(identity.get(key), -1)
            if gid >= 0:
                return gid, f"identity_evidence_{key}"
        for key in ("target_global_runtime_id", "target_global_runtime_id_int", "target_source_global_id_int", "target_global_id", "global_id", "global_person_id"):
            gid = _safe_gid_int(cand.get(key), -1)
            if gid >= 0:
                return gid, f"candidate_{key}"
        return -1, ""

    def _candidate_field_xy(self, cand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        identity = cand.get("target_identity_evidence") if isinstance(cand.get("target_identity_evidence"), dict) else {}
        for obj in (cand, identity):
            for key in ("target_local_xy", "local_xy", "fused_xy", "local_xy_raw", "field_xy"):
                xy = _xy(obj.get(key)) if isinstance(obj, dict) else None
                if xy is not None:
                    return xy
            phys = (obj.get("target_physical_coord") or obj.get("physical_coord")) if isinstance(obj, dict) else None
            if isinstance(phys, dict) and phys.get("ok", True):
                if phys.get("x_m") is not None and phys.get("y_m") is not None:
                    xy = _xy([phys.get("x_m"), phys.get("y_m")])
                    if xy is not None:
                        return xy
        return None

    def _track_xy(self, track: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        xy = track.get("_xy")
        if isinstance(xy, tuple) and len(xy) >= 2:
            return float(xy[0]), float(xy[1])
        for key in ("field_xy", "filtered_xy", "raw_xy", "measurement_xy", "association_xy"):
            xy = _xy(track.get(key))
            if xy is not None:
                return xy
        return None

    def _track_identity_evidence(
        self,
        cand: Dict[str, Any],
        track: Optional[Dict[str, Any]],
        reason: str,
        *,
        snapshot: Optional[Dict[str, Any]] = None,
        sync_delta: Optional[int] = None,
        local_id: Optional[int] = None,
        explicit_gid: int = -1,
    ) -> Dict[str, Any]:
        cam = self._candidate_camera(cand)
        cand_sync = self._candidate_sync_index(cand)
        out: Dict[str, Any] = {
            "schema_version": 1,
            "binding_reason": str(reason),
            "camera": str(cam),
            "camera_local_id": int(local_id if local_id is not None else self._candidate_local_id(cand)),
            "candidate_sync_index": int(cand_sync),
            "candidate_event_ts_us": _safe_int(cand.get("event_ts", cand.get("impact_event_ts_us")), 0),
            "candidate_wall_time": _safe_float(cand.get("wall_time"), 0.0),
            "stable_id_is_global": bool(cand.get("stable_id_is_global", False)),
            "strong": str(reason) not in {
                "stable_id_matches_global_id",
                "stable_id_global_fallback_disabled",
                "no_history_local_id_match",
                "no_target_global_id",
            },
        }
        if explicit_gid >= 0:
            out["explicit_target_global_runtime_id"] = int(explicit_gid)
        if track is not None:
            gid = _safe_gid_int(track.get("_gid", track.get("global_id")), -1)
            out.update({
                "target_global_runtime_id": int(gid),
                "target_global_player_id": str(track.get("global_player_id") or (f"P{gid:02d}" if gid >= 0 else "")),
                "track_age_ms": _safe_float(track.get("age_ms"), 0.0),
                "track_confidence": _safe_float(track.get("confidence"), 0.0),
                "track_source_cameras": list(track.get("source_cameras", [])) if isinstance(track.get("source_cameras"), list) else [],
                "track_local_ids": dict(track.get("local_ids", {})) if isinstance(track.get("local_ids"), dict) else {},
            })
        if snapshot is not None:
            snap_sync = _safe_int(snapshot.get("sync_index"), -1)
            loaded_ms = _safe_int(snapshot.get("loaded_ms"), _now_ms())
            out.update({
                "snapshot_sync_index": int(snap_sync),
                "snapshot_timestamp_ms": _safe_int(snapshot.get("timestamp_ms"), -1),
                "snapshot_age_ms": max(0, _now_ms() - int(loaded_ms)),
                "snapshot_track_count": len(snapshot.get("tracks", [])) if isinstance(snapshot.get("tracks"), list) else 0,
            })
        if sync_delta is not None:
            out["sync_delta"] = int(sync_delta)
        return out

    def _find_target_in_person_history(
        self,
        cand: Dict[str, Any],
        local_id: int,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        cam = self._candidate_camera(cand)
        cand_sync = self._candidate_sync_index(cand)
        max_sync_delta = max(0, int(getattr(self.args, "identity_evidence_max_sync_delta", 8)))
        max_age_ms = float(getattr(self.args, "identity_evidence_history_max_age_ms", 5000.0))
        now_ms = _now_ms()
        best: Optional[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = None
        for snap in reversed(self.person_history):
            loaded_ms = _safe_int(snap.get("loaded_ms"), now_ms)
            if max_age_ms >= 0.0 and now_ms - loaded_ms > max_age_ms:
                continue
            snap_sync = _safe_int(snap.get("sync_index"), -1)
            if cand_sync >= 0 and snap_sync >= 0:
                sync_delta = abs(int(snap_sync) - int(cand_sync))
                if sync_delta > max_sync_delta:
                    continue
            else:
                sync_delta = 999999
            tracks = snap.get("tracks", [])
            if not isinstance(tracks, list):
                continue
            for tr in tracks:
                if not isinstance(tr, dict):
                    continue
                local_ids = tr.get("local_ids", {})
                if isinstance(local_ids, dict) and cam and _safe_int(local_ids.get(cam), -999999) == int(local_id):
                    age = max(0, now_ms - loaded_ms)
                    item = (int(sync_delta), int(age), tr, snap)
                    if best is None or item[:2] < best[:2]:
                        best = item
        if best is None:
            return None, "no_history_local_id_match", self._track_identity_evidence(cand, None, "no_history_local_id_match", local_id=local_id)
        sync_delta, _age, tr, snap = best
        reason = "history_local_id_map"
        evidence = self._track_identity_evidence(cand, tr, reason, snapshot=snap, sync_delta=sync_delta, local_id=local_id)
        return tr, reason, evidence

    def _find_target_by_spatial_xy(
        self,
        cand: Dict[str, Any],
        persons: List[Dict[str, Any]],
        local_id: int,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not bool(getattr(self.args, "identity_spatial_fallback_enable", True)):
            return None, "spatial_fallback_disabled", self._track_identity_evidence(cand, None, "spatial_fallback_disabled", local_id=local_id)
        cand_xy = self._candidate_field_xy(cand)
        if cand_xy is None:
            return None, "candidate_field_xy_missing", self._track_identity_evidence(cand, None, "candidate_field_xy_missing", local_id=local_id)
        cam = self._candidate_camera(cand)
        cand_sync = self._candidate_sync_index(cand)
        max_dist = max(0.05, float(getattr(self.args, "identity_spatial_fallback_max_dist_m", 1.2)))
        min_gap = max(0.0, float(getattr(self.args, "identity_spatial_fallback_min_gap_m", 0.25)))
        max_sync_delta = max(0, int(getattr(self.args, "identity_spatial_fallback_max_sync_delta", 16)))
        max_age_ms = float(getattr(self.args, "identity_evidence_history_max_age_ms", 5000.0))
        now_ms = _now_ms()
        candidates: List[Tuple[float, int, Dict[str, Any], Optional[Dict[str, Any]], str, Optional[int]]] = []

        def consider(track: Dict[str, Any], snapshot: Optional[Dict[str, Any]], reason: str) -> None:
            if not isinstance(track, dict):
                return
            gid = _safe_gid_int(track.get("_gid", track.get("global_id")), -1)
            if gid < 0:
                return
            if cam:
                cams = track.get("source_cameras", [])
                local_ids = track.get("local_ids", {})
                has_cam = (isinstance(cams, list) and cam in [str(x) for x in cams]) or (
                    isinstance(local_ids, dict) and cam in local_ids
                )
                if not has_cam and bool(getattr(self.args, "identity_spatial_fallback_require_camera", True)):
                    return
            xy = self._track_xy(track)
            if xy is None:
                return
            sync_delta: Optional[int] = None
            if snapshot is not None:
                snap_sync = _safe_int(snapshot.get("sync_index"), -1)
                if cand_sync >= 0 and snap_sync >= 0:
                    sync_delta = abs(int(snap_sync) - int(cand_sync))
                    if sync_delta > max_sync_delta:
                        return
                loaded_ms = _safe_int(snapshot.get("loaded_ms"), now_ms)
                if max_age_ms >= 0.0 and now_ms - loaded_ms > max_age_ms:
                    return
            dist = math.hypot(float(cand_xy[0]) - float(xy[0]), float(cand_xy[1]) - float(xy[1]))
            candidates.append((float(dist), _safe_int(track.get("age_ms"), 0), track, snapshot, reason, sync_delta))

        for tr in persons:
            consider(tr, None, "current_spatial_xy_fallback")
        for snap in reversed(self.person_history):
            tracks = snap.get("tracks", [])
            if not isinstance(tracks, list):
                continue
            for tr in tracks:
                consider(tr, snap, "history_spatial_xy_fallback")

        if not candidates:
            return None, "no_spatial_xy_candidate", self._track_identity_evidence(cand, None, "no_spatial_xy_candidate", local_id=local_id)
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_dist, _age, best_track, best_snap, reason, sync_delta = candidates[0]
        second_dist = candidates[1][0] if len(candidates) > 1 else float("inf")
        if best_dist > max_dist:
            ev = self._track_identity_evidence(cand, None, "spatial_xy_too_far", local_id=local_id)
            ev.update({
                "candidate_field_xy": [round(float(cand_xy[0]), 4), round(float(cand_xy[1]), 4)],
                "spatial_best_distance_m": round(float(best_dist), 4),
                "spatial_max_distance_m": round(float(max_dist), 4),
            })
            return None, "spatial_xy_too_far", ev
        if math.isfinite(second_dist) and second_dist - best_dist < min_gap and best_dist > max_dist * 0.5:
            ev = self._track_identity_evidence(cand, None, "spatial_xy_ambiguous", local_id=local_id)
            ev.update({
                "candidate_field_xy": [round(float(cand_xy[0]), 4), round(float(cand_xy[1]), 4)],
                "spatial_best_distance_m": round(float(best_dist), 4),
                "spatial_second_distance_m": round(float(second_dist), 4),
                "spatial_min_gap_m": round(float(min_gap), 4),
            })
            return None, "spatial_xy_ambiguous", ev
        evidence = self._track_identity_evidence(
            cand,
            best_track,
            reason,
            snapshot=best_snap,
            sync_delta=sync_delta,
            local_id=local_id,
        )
        evidence.update({
            "candidate_field_xy": [round(float(cand_xy[0]), 4), round(float(cand_xy[1]), 4)],
            "spatial_match_distance_m": round(float(best_dist), 4),
            "spatial_second_distance_m": None if not math.isfinite(second_dist) else round(float(second_dist), 4),
            "spatial_max_distance_m": round(float(max_dist), 4),
        })
        return best_track, reason, evidence

    def _target_gid(self, cand: Dict[str, Any], persons: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        cam = self._candidate_camera(cand)
        local_id = self._candidate_local_id(cand)
        explicit, explicit_reason = self._candidate_explicit_runtime_gid(cand)
        if explicit >= 0:
            for tr in persons:
                if int(tr.get("_gid", -1)) == explicit:
                    return tr, "explicit_global_id", self._track_identity_evidence(cand, tr, explicit_reason or "explicit_global_id", local_id=local_id, explicit_gid=explicit)
            for snap in reversed(self.person_history):
                tracks = snap.get("tracks", [])
                if not isinstance(tracks, list):
                    continue
                for tr in tracks:
                    if int(tr.get("_gid", -1)) == explicit:
                        sync_delta = abs(_safe_int(snap.get("sync_index"), -1) - self._candidate_sync_index(cand)) if self._candidate_sync_index(cand) >= 0 and _safe_int(snap.get("sync_index"), -1) >= 0 else None
                        return tr, "explicit_global_id_history", self._track_identity_evidence(cand, tr, explicit_reason or "explicit_global_id_history", snapshot=snap, sync_delta=sync_delta, local_id=local_id, explicit_gid=explicit)
            target = {"_gid": int(explicit), "global_id": int(explicit), "local_ids": {}, "source_cameras": []}
            return target, "explicit_global_id_unverified", self._track_identity_evidence(cand, target, explicit_reason or "explicit_global_id_unverified", local_id=local_id, explicit_gid=explicit)

        target, reason, evidence = self._find_target_in_person_history(cand, local_id)
        if target is not None:
            return target, reason, evidence

        for tr in persons:
            local_ids = tr.get("local_ids", {})
            if isinstance(local_ids, dict) and cam and _safe_int(local_ids.get(cam), -999999) == local_id:
                return tr, "current_local_id_map", self._track_identity_evidence(cand, tr, "current_local_id_map", local_id=local_id)

        target, spatial_reason, spatial_evidence = self._find_target_by_spatial_xy(cand, persons, local_id)
        if target is not None:
            return target, spatial_reason, spatial_evidence

        if bool(getattr(self.args, "allow_stable_id_global_fallback", False)):
            for tr in persons:
                if int(tr.get("_gid", -1)) == local_id:
                    return tr, "stable_id_matches_global_id", self._track_identity_evidence(cand, tr, "stable_id_matches_global_id", local_id=local_id)
        return None, "no_target_global_id", evidence

    def _track_match_score(self, cand: Dict[str, Any], track: Dict[str, Any]) -> float:
        cam = self._candidate_camera(cand)
        bullet_id = self._candidate_bullet_id(cand)
        score = 0.0
        if cam and str(track.get("camera") or "") == cam:
            score += 0.45
        if bullet_id > 0 and _safe_int(track.get("bullet_id"), 0) == bullet_id:
            score += 0.45
        c_ts = _safe_int(cand.get("event_ts", cand.get("impact_event_ts_us", cand.get("point_ts_us"))), 0)
        t_ts = _safe_int(track.get("last_ts_us"), 0)
        if c_ts > 0 and t_ts > 0:
            dt_ms = abs(c_ts - t_ts) / 1000.0
            score += max(0.0, 0.20 * (1.0 - min(dt_ms, 300.0) / 300.0))
        return score

    def _best_bullet_track(self, cand: Dict[str, Any], tracks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for tr in tracks:
            xy = _xy(tr.get("field_xy"))
            direction = _xy(tr.get("direction_xy"))
            if xy is None or direction is None:
                continue
            s = self._track_match_score(cand, tr)
            if s > best_score:
                best = tr
                best_score = s
        if best_score < float(self.args.track_match_min_score):
            return None
        return best

    def _score_source(self, hit_xy: Tuple[float, float], direction: Tuple[float, float], target_gid: int, person: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
        gid = int(person.get("_gid", -1))
        if gid < 0 or gid == target_gid:
            return None
        pxy = person.get("_xy")
        if not isinstance(pxy, tuple):
            return None
        dx, dy = float(direction[0]), float(direction[1])
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        dx, dy = dx / norm, dy / norm
        vx = float(pxy[0]) - float(hit_xy[0])
        vy = float(pxy[1]) - float(hit_xy[1])
        # Shooter should lie behind the hit point along the inverse bullet direction.
        behind = vx * (-dx) + vy * (-dy)
        lateral_sq = max(0.0, vx * vx + vy * vy - behind * behind)
        lateral = math.sqrt(lateral_sq)
        if behind < float(self.args.source_min_behind_m):
            return None
        if behind > float(self.args.source_max_behind_m):
            return None
        lateral_scale = max(0.1, float(self.args.source_lateral_scale_m))
        behind_scale = max(0.1, float(self.args.source_behind_scale_m))
        lateral_score = math.exp(-lateral / lateral_scale)
        behind_score = math.exp(-max(0.0, behind - float(self.args.source_preferred_behind_m)) / behind_scale)
        confidence_bonus = max(0.0, min(1.0, _safe_float(person.get("confidence"), 0.5)))
        score = 0.65 * lateral_score + 0.25 * behind_score + 0.10 * confidence_bonus
        return score, {
            "source_candidate_lateral_m": round(lateral, 4),
            "source_candidate_behind_m": round(behind, 4),
            "source_candidate_score": round(score, 4),
        }

    def _event_key(self, cand: Dict[str, Any], target_gid: int, source_gid: Optional[int]) -> str:
        existing = str(cand.get("event_id") or "").strip()
        if existing:
            return existing
        cam = self._candidate_camera(cand) or "global"
        sync_id = _safe_int(cand.get("impact_sync_index", cand.get("mvs_sync_index", cand.get("person_frame_sync_index"))), 0)
        bullet_id = self._candidate_bullet_id(cand)
        return f"G-{cam}-{sync_id}-{bullet_id}-{target_gid}-{source_gid if source_gid is not None else 'unknown'}"

    def _direct_hit_confidence(self, cand: Dict[str, Any]) -> float:
        for key in ("confidence", "total_score", "space_score"):
            value = _safe_float(cand.get(key), float("nan"))
            if math.isfinite(value):
                return max(0.0, min(1.0, float(value)))
        return 0.72

    def _display_evidence(self, source_gid: int, rec: Optional[Dict[str, Any]], reason: str, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": 1,
            "source_global_runtime_id": int(source_gid),
            "binding_reason": str(reason),
        }
        if rec is not None:
            out.update({
                "display_slot_id": _safe_int(rec.get("_display_slot_id", rec.get("display_slot_id")), -1),
                "display_player_id": str(rec.get("display_player_id") or ""),
                "source_global_id": _safe_gid_int(rec.get("source_global_id"), int(source_gid)),
                "age_ms": _safe_float(rec.get("age_ms"), 0.0),
                "state": str(rec.get("state", "")),
            })
        if snapshot is not None:
            loaded_ms = _safe_int(snapshot.get("loaded_ms"), _now_ms())
            out.update({
                "snapshot_age_ms": max(0, _now_ms() - int(loaded_ms)),
                "snapshot_record_count": len(snapshot.get("records", [])) if isinstance(snapshot.get("records"), list) else 0,
            })
        return out

    def _normalize_display_record(self, source_gid: int, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(rec, dict):
            return None
        slot_id = _safe_int(rec.get("display_slot_id"), -1)
        if slot_id < 0:
            label = str(rec.get("display_player_id") or "").strip().upper()
            if label.startswith("P"):
                slot_id = _safe_int(label[1:], -1)
        if slot_id < 0:
            return None
        out = dict(rec)
        out["_display_slot_id"] = int(slot_id)
        out["_source_global_id_int"] = int(source_gid)
        return out

    def _load_display_records(self) -> List[Dict[str, Any]]:
        obj = _read_json(self.global_bev_status_path) or {}
        gp = obj.get("global_person", obj.get("global_person_overlay", {}))
        if not isinstance(gp, dict):
            self.stats["display_id_missing_global_person_status"] += 1
            return []
        records: List[Dict[str, Any]] = []
        by_gid = gp.get("display_id_by_source_global_id", {})
        if isinstance(by_gid, dict):
            for key, raw in by_gid.items():
                gid = _safe_gid_int(key, -1)
                if gid < 0 and isinstance(raw, dict):
                    gid = _safe_gid_int(raw.get("source_global_id"), -1)
                if gid < 0:
                    continue
                if isinstance(raw, dict):
                    norm = self._normalize_display_record(gid, raw)
                    if norm is not None:
                        records.append(norm)
                else:
                    slot = _safe_int(raw, -1)
                    if slot >= 0:
                        records.append({
                            "_display_slot_id": int(slot),
                            "display_slot_id": int(slot),
                            "display_player_id": f"P{slot:02d}",
                            "_source_global_id_int": int(gid),
                            "source_global_id": _gid_str(gid),
                        })
        tracks = gp.get("display_tracks", [])
        if isinstance(tracks, list):
            for item in tracks:
                if not isinstance(item, dict):
                    continue
                gid = _safe_gid_int(item.get("source_global_id"), -1)
                if gid < 0:
                    continue
                norm = self._normalize_display_record(gid, item)
                if norm is not None and not any(_safe_int(r.get("_source_global_id_int"), -1) == gid for r in records):
                    records.append(norm)
        snapshot = {
            "loaded_ms": _now_ms(),
            "records": [dict(r) for r in records],
        }
        self.display_history.append(snapshot)
        return records

    def _lookup_global_bev_display_id(self, source_gid: int) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        reserved_id = _safe_int(getattr(self.args, "reserved_virtual_global_id", -1), -1)
        records = self._load_display_records()
        max_age = float(getattr(self.args, "global_bev_display_max_age_ms", 1200.0))
        for rec in records:
            if _safe_int(rec.get("_source_global_id_int"), -1) != int(source_gid):
                continue
            age = _safe_float(rec.get("age_ms"), 0.0)
            if max_age >= 0.0 and age > max_age:
                self.stats["display_id_lookup_stale"] += 1
                return None, "display_mapping_stale", self._display_evidence(source_gid, rec, "display_mapping_stale")
            slot_id = _safe_int(rec.get("_display_slot_id"), -1)
            if reserved_id >= 0 and slot_id == reserved_id:
                self.stats["display_id_reserved_virtual_target"] += 1
                return None, "display_slot_reserved_virtual_id", self._display_evidence(source_gid, rec, "display_slot_reserved_virtual_id")
            self.stats["display_id_lookup_hit"] += 1
            return rec, "global_bev_display", self._display_evidence(source_gid, rec, "global_bev_display")

        max_hist_age_ms = float(getattr(self.args, "display_history_max_age_ms", 5000.0))
        now_ms = _now_ms()
        for snap in reversed(self.display_history):
            loaded_ms = _safe_int(snap.get("loaded_ms"), now_ms)
            if max_hist_age_ms >= 0.0 and now_ms - loaded_ms > max_hist_age_ms:
                continue
            records = snap.get("records", [])
            if not isinstance(records, list):
                continue
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                if _safe_int(rec.get("_source_global_id_int"), -1) != int(source_gid):
                    continue
                slot_id = _safe_int(rec.get("_display_slot_id"), -1)
                if reserved_id >= 0 and slot_id == reserved_id:
                    self.stats["display_id_reserved_virtual_target"] += 1
                    return None, "display_slot_reserved_virtual_id", self._display_evidence(source_gid, rec, "display_slot_reserved_virtual_id", snap)
                self.stats["display_id_history_lookup_hit"] += 1
                return dict(rec), "global_bev_display_history", self._display_evidence(source_gid, rec, "global_bev_display_history", snap)
        self.stats["display_id_lookup_miss"] += 1
        return None, "display_mapping_missing", self._display_evidence(source_gid, None, "display_mapping_missing")

    def _process_direct_candidate(self, cand: Dict[str, Any], persons: List[Dict[str, Any]]) -> Dict[str, Any]:
        target, target_reason, target_evidence = self._target_gid(cand, persons)
        if target is None:
            return self._reject(cand, "no_target_global_id", {
                "target_reason": target_reason,
                "target_identity_evidence": target_evidence,
            })
        source_target_gid = int(target["_gid"])
        output_target_gid = int(source_target_gid)
        output_target_reason = str(target_reason)
        target_display: Dict[str, Any] = {}
        target_display_evidence: Dict[str, Any] = {}
        target_id_source = str(getattr(self.args, "target_id_source", "global_runtime") or "global_runtime")
        if target_id_source == "global_bev_display":
            display_rec, display_reason, target_display_evidence = self._lookup_global_bev_display_id(source_target_gid)
            if display_rec is not None:
                output_target_gid = int(display_rec.get("_display_slot_id", source_target_gid))
                output_target_reason = str(display_reason)
                target_display = display_rec
            else:
                return self._reject(
                    cand,
                    "no_global_bev_display_id",
                    {
                        "target_global_id_int": source_target_gid,
                        "target_global_id": _gid_str(source_target_gid),
                        "target_reason": str(target_reason),
                        "display_reason": str(display_reason),
                        "target_id_source": str(target_id_source),
                        "target_identity_evidence": target_evidence,
                        "target_display_evidence": target_display_evidence,
                    },
                )
        source_gid = int(getattr(self.args, "default_source_global_id", 99))
        confidence = self._direct_hit_confidence(cand)
        event_id = self._event_key(cand, output_target_gid, source_gid)
        publishable = confidence >= float(self.args.score_threshold)
        payload: Dict[str, Any] = {
            "type": "damage_attribution",
            "schema_version": 1,
            "source": "damage_attribution_service",
            "mode": str(self.args.mode),
            "attribution_source": "direct_hit_global_id",
            "event_id": event_id,
            "camera": self._candidate_camera(cand),
            "id_semantics": "global_bev_display_id" if target_id_source == "global_bev_display" else "global_runtime_id",
            "source_global_id": _gid_str(source_gid),
            "source_global_id_int": int(source_gid),
            "target_global_id": _gid_str(output_target_gid),
            "target_global_id_int": int(output_target_gid),
            "target_id_source": str(target_id_source),
            "target_source_global_id": _gid_str(source_target_gid),
            "target_source_global_id_int": int(source_target_gid),
            "target_display_player_id": str(target_display.get("display_player_id", "")),
            "target_display_slot_id": _safe_int(target_display.get("_display_slot_id"), -1),
            "target_identity_evidence": target_evidence,
            "target_display_evidence": target_display_evidence,
            "confidence": round(float(confidence), 4),
            "publishable": bool(publishable),
            "reject_reason": "" if publishable else "below_score_threshold",
            "score_threshold": float(self.args.score_threshold),
            "target_reason": str(output_target_reason),
            "target_global_runtime_reason": str(target_reason),
            "candidate_total_score": _safe_float(cand.get("total_score", cand.get("confidence")), confidence),
            "default_source_global_id": int(source_gid),
            "wall_time": time.time(),
            "candidate": {
                "bullet_id": self._candidate_bullet_id(cand),
                "stable_id": _safe_int(cand.get("stable_id", cand.get("person_id")), -1),
                "event_ts": _safe_int(cand.get("event_ts", cand.get("impact_event_ts_us")), 0),
                "match_method": str(cand.get("match_method", cand.get("geometry_mode", "")) or ""),
                "judge_chain": str(cand.get("judge_chain", "")),
                "judge_mode": str(cand.get("judge_mode", "")),
                "target_camera": str(cand.get("target_camera", self._candidate_camera(cand))),
                "target_camera_local_id": self._candidate_local_id(cand),
                "target_person_sync_index": self._candidate_sync_index(cand),
            },
        }
        self.stats["direct_hit_events"] += 1
        self.stats["attributed_events"] += 1
        if publishable:
            self.stats["publishable_events"] += 1
        else:
            self.stats["below_threshold_events"] += 1
        return payload

    def _process_candidate(self, cand: Dict[str, Any], persons: List[Dict[str, Any]], bullet_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.direct_hit_global_id_mode:
            return self._process_direct_candidate(cand, persons)
        target, target_reason, target_evidence = self._target_gid(cand, persons)
        if target is None:
            return self._reject(cand, "no_target_global_id", {
                "target_reason": target_reason,
                "target_identity_evidence": target_evidence,
            })
        target_gid = int(target["_gid"])
        bullet_track = self._best_bullet_track(cand, bullet_tracks)
        if bullet_track is None:
            return self._reject(cand, "no_matching_global_bullet_track", {"target_global_id_int": target_gid, "target_reason": target_reason})
        hit_xy = _xy(bullet_track.get("field_xy"))
        direction = _xy(bullet_track.get("direction_xy"))
        if hit_xy is None or direction is None:
            return self._reject(cand, "bad_bullet_track_geometry", {"target_global_id_int": target_gid})
        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for person in persons:
            s = self._score_source(hit_xy, direction, target_gid, person)
            if s is not None:
                score, detail = s
                scored.append((score, person, detail))
        if not scored:
            return self._reject(cand, "no_source_candidate", {"target_global_id_int": target_gid})
        scored.sort(key=lambda x: x[0], reverse=True)
        source_score, source, source_detail = scored[0]
        source_gid = int(source["_gid"])
        cand_score = max(0.0, min(1.0, _safe_float(cand.get("total_score", cand.get("confidence")), 0.65)))
        track_score = max(0.0, min(1.0, _safe_float(bullet_track.get("source_record", {}).get("confidence"), 1.0)))
        confidence = max(0.0, min(1.0, 0.50 * source_score + 0.30 * cand_score + 0.20 * track_score))
        event_id = self._event_key(cand, target_gid, source_gid)
        publishable = confidence >= float(self.args.score_threshold)
        payload: Dict[str, Any] = {
            "type": "damage_attribution",
            "schema_version": 1,
            "source": "damage_attribution_service",
            "mode": str(self.args.mode),
            "event_id": event_id,
            "camera": self._candidate_camera(cand),
            "id_semantics": "global_runtime_id",
            "source_global_id": _gid_str(source_gid),
            "source_global_id_int": int(source_gid),
            "target_global_id": _gid_str(target_gid),
            "target_global_id_int": int(target_gid),
            "confidence": round(float(confidence), 4),
            "publishable": bool(publishable),
            "reject_reason": "" if publishable else "below_score_threshold",
            "score_threshold": float(self.args.score_threshold),
            "bullet_track_id": str(bullet_track.get("track_id", "")),
            "bullet_field_xy": list(hit_xy),
            "bullet_direction_xy": list(direction),
            "target_reason": str(target_reason),
            "target_identity_evidence": target_evidence,
            "candidate_total_score": cand_score,
            "track_confidence": track_score,
            "wall_time": time.time(),
            "candidate": {
                "bullet_id": self._candidate_bullet_id(cand),
                "stable_id": _safe_int(cand.get("stable_id", cand.get("person_id")), -1),
                "event_ts": _safe_int(cand.get("event_ts", cand.get("impact_event_ts_us")), 0),
                "match_method": str(cand.get("match_method", cand.get("geometry_mode", "")) or ""),
                "target_camera": str(cand.get("target_camera", self._candidate_camera(cand))),
                "target_camera_local_id": self._candidate_local_id(cand),
                "target_person_sync_index": self._candidate_sync_index(cand),
            },
        }
        payload.update(source_detail)
        self.stats["attributed_events"] += 1
        if publishable:
            self.stats["publishable_events"] += 1
        else:
            self.stats["below_threshold_events"] += 1
        return payload

    def _reject(self, cand: Dict[str, Any], reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": "damage_attribution",
            "schema_version": 1,
            "source": "damage_attribution_service",
            "mode": str(self.args.mode),
            "event_id": self._event_key(cand, -1, None),
            "camera": self._candidate_camera(cand),
            "id_semantics": "global_runtime_id",
            "publishable": False,
            "reject_reason": reason,
            "confidence": 0.0,
            "wall_time": time.time(),
            "candidate": {
                "bullet_id": self._candidate_bullet_id(cand),
                "stable_id": _safe_int(cand.get("stable_id", cand.get("person_id")), -1),
                "event_ts": _safe_int(cand.get("event_ts", cand.get("impact_event_ts_us")), 0),
                "target_camera": str(cand.get("target_camera", self._candidate_camera(cand))),
                "target_camera_local_id": self._candidate_local_id(cand),
                "target_person_sync_index": self._candidate_sync_index(cand),
            },
        }
        if extra:
            payload.update(extra)
        self.stats[f"reject_{reason}"] += 1
        return payload

    def _dedup(self, event: Dict[str, Any]) -> bool:
        key = str(event.get("event_id") or "")
        if not key:
            return False
        now = time.time()
        ttl = max(1.0, float(self.args.event_id_ttl_sec))
        expired = [k for k, t in self.event_seen.items() if now - t > ttl]
        for k in expired[:1024]:
            self.event_seen.pop(k, None)
        if key in self.event_seen:
            self.stats["duplicate_events"] += 1
            return True
        self.event_seen[key] = now
        return False

    def _write_event(self, event: Dict[str, Any]) -> None:
        if self._dedup(event):
            return
        self.latest_event = dict(event)
        self.out_fp.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.stats["output_events"] += 1

    def _publish_status(self, persons: List[Dict[str, Any]], tracks: List[Dict[str, Any]]) -> None:
        latest = {
            "schema_version": 1,
            "msg_type": "damage_attribution_latest",
            "mode": str(self.args.mode),
            "timestamp_ms": _now_ms(),
            "latest_event": dict(self.latest_event),
        }
        status = {
            "schema_version": 1,
            "source": "damage_attribution_service",
            "mode": str(self.args.mode),
            "hit_jsonl": str(self.hit_path),
            "person_latest_json": str(self.person_latest_path),
            "track_latest_json": str(self.track_latest_path),
            "global_bev_status_json": str(self.global_bev_status_path),
            "events_jsonl": str(self.events_path),
            "attribution_source": str(getattr(self.args, "attribution_source", "global_bullet")),
            "target_id_source": str(getattr(self.args, "target_id_source", "global_runtime")),
            "default_source_global_id": int(getattr(self.args, "default_source_global_id", 99)),
            "reserved_virtual_global_id": int(getattr(self.args, "reserved_virtual_global_id", -1)),
            "identity_evidence": {
                "person_history_snapshots": len(self.person_history),
                "display_history_snapshots": len(self.display_history),
                "identity_evidence_max_sync_delta": int(getattr(self.args, "identity_evidence_max_sync_delta", 8)),
                "identity_evidence_history_max_age_ms": float(getattr(self.args, "identity_evidence_history_max_age_ms", 5000.0)),
                "identity_spatial_fallback_enable": bool(getattr(self.args, "identity_spatial_fallback_enable", True)),
                "identity_spatial_fallback_max_dist_m": float(getattr(self.args, "identity_spatial_fallback_max_dist_m", 1.2)),
                "identity_spatial_fallback_min_gap_m": float(getattr(self.args, "identity_spatial_fallback_min_gap_m", 0.25)),
                "identity_spatial_fallback_max_sync_delta": int(getattr(self.args, "identity_spatial_fallback_max_sync_delta", 16)),
                "display_history_max_age_ms": float(getattr(self.args, "display_history_max_age_ms", 5000.0)),
                "allow_stable_id_global_fallback": bool(getattr(self.args, "allow_stable_id_global_fallback", False)),
            },
            "person_track_count": len(persons),
            "bullet_track_count": len(tracks),
            "stats": dict(self.stats),
            "tail_error": str(self.hit_tailer.error),
            "timestamp_ms": _now_ms(),
        }
        _atomic_write_json(self.latest_path, latest)
        _atomic_write_json(self.status_path, status)

    def run(self) -> None:
        print(
            f"[damage_attribution] start mode={self.args.mode} hit={self.hit_path} events={self.events_path}",
            flush=True,
        )
        next_status = 0.0
        persons: List[Dict[str, Any]] = []
        bullet_tracks: List[Dict[str, Any]] = []
        while True:
            if not self.mode_enabled:
                self._publish_status(persons, bullet_tracks)
                time.sleep(max(0.1, float(self.args.interval_sec)))
                continue
            persons = self._load_person_tracks()
            bullet_tracks = self._load_bullet_tracks()
            candidates = self.hit_tailer.read_new(max_lines=int(self.args.max_lines_per_poll))
            self.stats["input_candidates"] += len(candidates)
            for cand in candidates:
                event = self._process_candidate(cand, persons, bullet_tracks)
                self._write_event(event)
            now = time.time()
            if now >= next_status:
                self._publish_status(persons, bullet_tracks)
                next_status = now + max(0.1, float(self.args.status_interval_sec))
            time.sleep(max(0.01, float(self.args.interval_sec)))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Attribute hit candidates to runtime global source IDs")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--mode", choices=["disabled", "shadow_no_publish", "shadow", "publish", "enforce"], default="disabled")
    ap.add_argument("--hit-jsonl", default="hit_candidate_all.jsonl")
    ap.add_argument("--track-jsonl", default="global_bullet_tracks.jsonl")
    ap.add_argument("--track-latest-json", default="global_bullet_fusion_latest.json")
    ap.add_argument("--person-latest-json", default="global_person_latest.json")
    ap.add_argument("--global-bev-status-json", default="global_bev_status.json")
    ap.add_argument("--events-jsonl", default="damage_attribution_events.jsonl")
    ap.add_argument("--latest-json", default="damage_attribution_latest.json")
    ap.add_argument("--status-json", default="damage_attribution_status.json")
    ap.add_argument("--attribution-source", choices=["global_bullet", "direct_hit_global_id"], default="global_bullet")
    ap.add_argument("--target-id-source", choices=["global_runtime", "global_bev_display"], default="global_runtime")
    ap.add_argument("--global-bev-display-max-age-ms", type=float, default=1200.0)
    ap.add_argument("--default-source-global-id", type=int, default=99)
    ap.add_argument("--score-threshold", type=float, default=0.45)
    ap.add_argument("--interval-sec", type=float, default=0.08)
    ap.add_argument("--status-interval-sec", type=float, default=1.0)
    ap.add_argument("--person-max-age-ms", type=float, default=800.0)
    ap.add_argument("--person-history-max-snapshots", type=int, default=2048)
    ap.add_argument("--identity-evidence-max-sync-delta", type=int, default=8)
    ap.add_argument("--identity-evidence-history-max-age-ms", type=float, default=5000.0)
    ap.add_argument("--identity-spatial-fallback-enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--identity-spatial-fallback-require-camera", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--identity-spatial-fallback-max-dist-m", type=float, default=1.2)
    ap.add_argument("--identity-spatial-fallback-min-gap-m", type=float, default=0.25)
    ap.add_argument("--identity-spatial-fallback-max-sync-delta", type=int, default=16)
    ap.add_argument("--display-history-max-snapshots", type=int, default=2048)
    ap.add_argument("--display-history-max-age-ms", type=float, default=5000.0)
    ap.add_argument("--allow-stable-id-global-fallback", action="store_true", default=False)
    ap.add_argument("--reserved-virtual-global-id", type=int, default=-1)
    ap.add_argument("--track-match-min-score", type=float, default=0.45)
    ap.add_argument("--source-min-behind-m", type=float, default=0.4)
    ap.add_argument("--source-preferred-behind-m", type=float, default=4.0)
    ap.add_argument("--source-max-behind-m", type=float, default=35.0)
    ap.add_argument("--source-lateral-scale-m", type=float, default=1.2)
    ap.add_argument("--source-behind-scale-m", type=float, default=9.0)
    ap.add_argument("--event-id-ttl-sec", type=float, default=60.0)
    ap.add_argument("--max-lines-per-poll", type=int, default=1000)
    ap.add_argument("--read-existing", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    DamageAttributionService(args).run()


if __name__ == "__main__":
    main()
