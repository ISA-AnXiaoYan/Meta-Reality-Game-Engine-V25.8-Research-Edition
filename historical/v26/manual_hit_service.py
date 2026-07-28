#!/usr/bin/env python3
"""Publish operator-confirmed manual hit commands to the game-event bridge.

The service is intentionally independent from automatic hit judgement.  Preview
clicks append small command records, this sidecar resolves the clicked person to
the current Global BEV display ID when possible, then appends a publishable hit
event to damage_attribution_events.jsonl.  game_event_bridge_service remains the
only process that talks to the game manager.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_gid_int(value: Any, default: int = -1) -> int:
    if isinstance(value, str):
        raw = value.strip().upper()
        if raw.startswith("G") or raw.startswith("P"):
            raw = raw[1:]
        return _safe_int(raw, default)
    return _safe_int(value, default)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _gid_str(gid: int) -> str:
    return f"G{int(gid):04d}"


def _display_player_id(slot_id: int) -> str:
    return f"P{int(slot_id):02d}"


def _resolve_path(sync_root: Path, project_root: Path, value: str) -> Path:
    p = Path(str(value or ""))
    if p.is_absolute():
        return p
    raw = str(value or "").replace("\\", "/")
    if raw.startswith("sync_ipc/"):
        return project_root / p
    return sync_root / p


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _xy(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _safe_float(value[0])
        y = _safe_float(value[1])
        if math.isfinite(x) and math.isfinite(y):
            return float(x), float(y)
    if isinstance(value, dict):
        x = _safe_float(value.get("x", value.get("X")))
        y = _safe_float(value.get("y", value.get("Y")))
        if math.isfinite(x) and math.isfinite(y):
            return float(x), float(y)
    return None


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class JsonlTailer:
    def __init__(self, path: Path, read_existing: bool = False) -> None:
        self.path = Path(path)
        self.offset = 0
        self.inode: Optional[Tuple[int, int]] = None
        self.error = ""
        if not read_existing:
            try:
                self.offset = self.path.stat().st_size
            except FileNotFoundError:
                self.offset = 0
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    def read_new(self, max_lines: int = 512) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self.error = ""
            return out
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return out
        inode = (int(getattr(st, "st_ino", 0)), int(st.st_size))
        if self.inode is not None and (inode[0] != self.inode[0] or st.st_size < self.offset):
            self.offset = 0
        self.inode = inode
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fp:
                fp.seek(self.offset)
                for idx, line in enumerate(fp):
                    if idx >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        continue
                self.offset = fp.tell()
            self.error = ""
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        return out


class ManualHitService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.sync_root = Path(args.sync_root).resolve()
        self.command_path = _resolve_path(self.sync_root, self.project_root, args.command_jsonl)
        self.confirmed_path = _resolve_path(self.sync_root, self.project_root, args.confirmed_jsonl)
        self.events_path = _resolve_path(self.sync_root, self.project_root, args.events_jsonl)
        self.latest_path = _resolve_path(self.sync_root, self.project_root, args.latest_json)
        self.status_path = _resolve_path(self.sync_root, self.project_root, args.status_json)
        self.control_path = _resolve_path(self.sync_root, self.project_root, args.control_json)
        self.global_bev_status_path = _resolve_path(self.sync_root, self.project_root, args.global_bev_status_json)
        self.person_latest_path = _resolve_path(self.sync_root, self.project_root, args.person_latest_json)
        self.command_tailer = JsonlTailer(self.command_path, bool(args.read_existing))
        self.stats = Counter()
        self.latest_result: Dict[str, Any] = {}
        self.latest_command: Dict[str, Any] = {}
        self.latest_event: Dict[str, Any] = {}
        self.recent_by_target: Dict[int, float] = {}
        self.last_error = ""
        self.control_seq = 0
        self.control_mtime_ns = 0
        self.control: Dict[str, Any] = {
            "enabled": True,
            "source_global_id": int(args.source_global_id),
            "dedup_ms": float(args.dedup_ms),
            "target_max_age_ms": float(args.target_max_age_ms),
            "target_match_max_dist_m": float(args.target_match_max_dist_m),
        }
        self.command_path.parent.mkdir(parents=True, exist_ok=True)
        self.confirmed_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.confirmed_fp = self.confirmed_path.open("a", encoding="utf-8", buffering=1)
        self.events_fp = self.events_path.open("a", encoding="utf-8", buffering=1)

    @property
    def mode_enabled(self) -> bool:
        return str(self.args.mode) not in {"disabled", "off"}

    def _poll_control(self) -> None:
        try:
            st = self.control_path.stat()
        except FileNotFoundError:
            return
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return
        if int(st.st_mtime_ns) <= int(self.control_mtime_ns):
            return
        try:
            obj = _read_json(self.control_path)
            if obj:
                self.control.update({
                    "enabled": _as_bool(obj.get("enabled", obj.get("manual_hit_enabled")), True),
                    "source_global_id": _safe_gid_int(obj.get("source_global_id", obj.get("source_global_id_int")), int(self.args.source_global_id)),
                    "dedup_ms": _safe_float(obj.get("dedup_ms"), float(self.args.dedup_ms)),
                    "target_max_age_ms": _safe_float(obj.get("target_max_age_ms"), float(self.args.target_max_age_ms)),
                    "target_match_max_dist_m": _safe_float(obj.get("target_match_max_dist_m"), float(self.args.target_match_max_dist_m)),
                })
                self.control_seq = _safe_int(obj.get("seq"), self.control_seq + 1)
            self.control_mtime_ns = int(st.st_mtime_ns)
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _display_tracks(self) -> List[Dict[str, Any]]:
        status = _read_json(self.global_bev_status_path)
        candidates: List[Any] = []
        for key in ("global_person", "global_person_overlay", "global_person_diag"):
            val = status.get(key)
            if isinstance(val, dict):
                candidates.append(val)
        for container in candidates:
            tracks = container.get("display_tracks", container.get("tracks", []))
            if isinstance(tracks, list):
                return [dict(x) for x in tracks if isinstance(x, dict)]
        return []

    def _runtime_tracks(self) -> List[Dict[str, Any]]:
        latest = _read_json(self.person_latest_path)
        tracks = latest.get("tracks", latest.get("persons", []))
        if not isinstance(tracks, list):
            return []
        return [dict(x) for x in tracks if isinstance(x, dict)]

    def _target_from_display_track(self, track: Dict[str, Any], reason: str, distance_m: float = -1.0) -> Optional[Dict[str, Any]]:
        slot_id = _safe_int(track.get("display_slot_id", track.get("_display_slot_id")), -1)
        display_player_id = str(track.get("display_player_id") or (_display_player_id(slot_id) if slot_id >= 0 else ""))
        source_gid = _safe_gid_int(track.get("source_global_id", track.get("global_id")), -1)
        if slot_id >= 0:
            return {
                "target_global_id_int": int(slot_id),
                "target_global_id": _gid_str(slot_id),
                "target_display_slot_id": int(slot_id),
                "target_display_player_id": display_player_id,
                "target_source_global_id_int": int(source_gid),
                "target_source_global_id": _gid_str(source_gid) if source_gid >= 0 else "",
                "target_id_source": "global_bev_display",
                "id_semantics": "global_bev_display_id",
                "resolve_reason": reason,
                "resolve_distance_m": float(distance_m),
                "target_track": track,
            }
        runtime_gid = _safe_gid_int(track.get("global_id", track.get("source_global_id")), -1)
        if runtime_gid >= 0:
            return {
                "target_global_id_int": int(runtime_gid),
                "target_global_id": _gid_str(runtime_gid),
                "target_display_slot_id": -1,
                "target_display_player_id": display_player_id or _display_player_id(runtime_gid),
                "target_source_global_id_int": int(runtime_gid),
                "target_source_global_id": _gid_str(runtime_gid),
                "target_id_source": "global_runtime",
                "id_semantics": "global_runtime_id",
                "resolve_reason": reason,
                "resolve_distance_m": float(distance_m),
                "target_track": track,
            }
        return None

    def _target_from_runtime_track(self, track: Dict[str, Any], reason: str, distance_m: float = -1.0) -> Optional[Dict[str, Any]]:
        gid = _safe_gid_int(track.get("global_id"), -1)
        if gid < 0:
            return None
        return {
            "target_global_id_int": int(gid),
            "target_global_id": _gid_str(gid),
            "target_display_slot_id": -1,
            "target_display_player_id": str(track.get("global_player_id") or _display_player_id(gid)),
            "target_source_global_id_int": int(gid),
            "target_source_global_id": _gid_str(gid),
            "target_id_source": "global_runtime",
            "id_semantics": "global_runtime_id",
            "resolve_reason": reason,
            "resolve_distance_m": float(distance_m),
            "target_track": track,
        }

    def _match_by_local_id(self, cmd: Dict[str, Any], tracks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cam = str(cmd.get("camera") or cmd.get("cam") or "").strip()
        if not cam:
            return None
        lid = _safe_int(cmd.get("stable_id", cmd.get("person_id")), -1)
        if lid < 0:
            return None
        for track in tracks:
            local_ids = track.get("local_ids")
            if isinstance(local_ids, dict) and _safe_int(local_ids.get(cam), -999999) == lid:
                return self._target_from_display_track(track, "global_bev_display_local_id")
        for track in self._runtime_tracks():
            local_ids = track.get("local_ids")
            if isinstance(local_ids, dict) and _safe_int(local_ids.get(cam), -999999) == lid:
                return self._target_from_runtime_track(track, "global_runtime_local_id")
        return None

    def _match_by_xy(self, cmd: Dict[str, Any], tracks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        xy = _xy(cmd.get("field_xy")) or _xy(cmd.get("local_xy")) or _xy(cmd.get("physical_coord"))
        if xy is None:
            return None
        max_dist = max(0.1, float(self.control.get("target_match_max_dist_m", self.args.target_match_max_dist_m)))
        best_track: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for track in tracks:
            txy = _xy(track.get("field_xy")) or _xy(track.get("filtered_xy")) or _xy(track.get("raw_xy"))
            if txy is None:
                continue
            dd = _dist(xy, txy)
            if dd < best_dist:
                best_dist = dd
                best_track = track
        if best_track is not None and best_dist <= max_dist:
            return self._target_from_display_track(best_track, "global_bev_display_nearest_xy", best_dist)
        best_track = None
        best_dist = float("inf")
        for track in self._runtime_tracks():
            txy = _xy(track.get("field_xy")) or _xy(track.get("filtered_xy")) or _xy(track.get("raw_xy"))
            if txy is None:
                continue
            dd = _dist(xy, txy)
            if dd < best_dist:
                best_dist = dd
                best_track = track
        if best_track is not None and best_dist <= max_dist:
            return self._target_from_runtime_track(best_track, "global_runtime_nearest_xy", best_dist)
        return None

    def _resolve_target(self, cmd: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        explicit_sem = str(cmd.get("target_id_semantics") or "").strip().lower()
        explicit_slot = _safe_int(cmd.get("target_display_slot_id"), -1)
        if explicit_slot >= 0:
            return {
                "target_global_id_int": int(explicit_slot),
                "target_global_id": _gid_str(explicit_slot),
                "target_display_slot_id": int(explicit_slot),
                "target_display_player_id": str(cmd.get("target_display_player_id") or _display_player_id(explicit_slot)),
                "target_source_global_id_int": _safe_gid_int(cmd.get("target_source_global_id_int"), -1),
                "target_source_global_id": str(cmd.get("target_source_global_id") or ""),
                "target_id_source": "global_bev_display",
                "id_semantics": "global_bev_display_id",
                "resolve_reason": "command_explicit_display_slot",
                "resolve_distance_m": -1.0,
                "target_track": {},
            }, ""
        explicit_gid = _safe_gid_int(cmd.get("target_global_id_int", cmd.get("target_global_id")), -1)
        if explicit_gid >= 0 and explicit_sem in {"global_runtime_id", "global_bev_display_id"}:
            source = "global_bev_display" if explicit_sem == "global_bev_display_id" else "global_runtime"
            return {
                "target_global_id_int": int(explicit_gid),
                "target_global_id": _gid_str(explicit_gid),
                "target_display_slot_id": int(explicit_gid) if source == "global_bev_display" else -1,
                "target_display_player_id": str(cmd.get("target_display_player_id") or _display_player_id(explicit_gid)),
                "target_source_global_id_int": _safe_gid_int(cmd.get("target_source_global_id_int"), explicit_gid),
                "target_source_global_id": str(cmd.get("target_source_global_id") or _gid_str(explicit_gid)),
                "target_id_source": source,
                "id_semantics": explicit_sem,
                "resolve_reason": "command_explicit_global_id",
                "resolve_distance_m": -1.0,
                "target_track": {},
            }, ""
        display_tracks = self._display_tracks()
        target = self._match_by_local_id(cmd, display_tracks)
        if target is not None:
            return target, ""
        target = self._match_by_xy(cmd, display_tracks)
        if target is not None:
            return target, ""
        return None, "target_unresolved"

    def _dedup(self, target_id: int) -> bool:
        now = time.time() * 1000.0
        ttl = max(0.0, float(self.control.get("dedup_ms", self.args.dedup_ms)))
        for key, ts in list(self.recent_by_target.items()):
            if now - ts > max(ttl, 1000.0) * 4.0:
                self.recent_by_target.pop(key, None)
        if ttl > 0 and target_id in self.recent_by_target and now - self.recent_by_target[target_id] < ttl:
            return True
        self.recent_by_target[target_id] = now
        return False

    def _reject(self, cmd: Dict[str, Any], reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = {
            "schema_version": 1,
            "type": "manual_hit_confirm",
            "source": "manual_hit_service",
            "accepted": False,
            "publishable": False,
            "reject_reason": str(reason),
            "command": cmd,
            "wall_time": time.time(),
            "wall_time_us": _now_us(),
        }
        if extra:
            rec.update(extra)
        self.confirmed_fp.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.latest_result = rec
        self.stats[f"reject_{reason}"] += 1
        return rec

    def _command_event_source(self, cmd: Dict[str, Any]) -> Tuple[str, str, str]:
        raw = str(cmd.get("click_source") or cmd.get("source") or "").strip().lower()
        if "global_bev" in raw:
            return "MANUAL-GLOBAL-BEV", "manual_global_bev_click", "operator_global_bev_click"
        return "MANUAL-PREVIEW", "manual_preview_click", "operator_preview_click"

    def _accept(self, cmd: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        source_gid = _safe_gid_int(self.control.get("source_global_id"), int(self.args.source_global_id))
        target_gid = _safe_int(target.get("target_global_id_int"), -1)
        if self._dedup(target_gid):
            return self._reject(cmd, "dedup_target_cooldown", {"target_global_id_int": target_gid})
        command_id = str(cmd.get("command_id") or cmd.get("event_id") or f"CMD-{_now_us()}")
        event_prefix, event_source, manual_authority = self._command_event_source(cmd)
        event_id = f"{event_prefix}-{_now_us()}-{source_gid}-{target_gid}"
        event = {
            "schema_version": 1,
            "type": "damage_attribution",
            "source": event_source,
            "manual": True,
            "manual_authority": manual_authority,
            "mode": str(self.args.mode),
            "attribution_source": event_source,
            "event_id": event_id,
            "camera": str(cmd.get("camera") or "preview"),
            "id_semantics": str(target.get("id_semantics") or "global_bev_display_id"),
            "source_global_id": _gid_str(source_gid),
            "source_global_id_int": int(source_gid),
            "target_global_id": str(target.get("target_global_id") or _gid_str(target_gid)),
            "target_global_id_int": int(target_gid),
            "target_id_source": str(target.get("target_id_source") or "global_bev_display"),
            "target_source_global_id": str(target.get("target_source_global_id") or ""),
            "target_source_global_id_int": _safe_int(target.get("target_source_global_id_int"), -1),
            "target_display_player_id": str(target.get("target_display_player_id") or ""),
            "target_display_slot_id": _safe_int(target.get("target_display_slot_id"), -1),
            "target_identity_evidence": {
                "manual_command_id": command_id,
                "manual_click_xy": cmd.get("click_xy"),
                "manual_camera": cmd.get("camera"),
                "manual_stable_id": cmd.get("stable_id"),
                "manual_person_id": cmd.get("person_id"),
                "manual_field_xy": cmd.get("field_xy") or cmd.get("local_xy"),
                "resolve_reason": target.get("resolve_reason"),
                "resolve_distance_m": target.get("resolve_distance_m"),
            },
            "target_display_evidence": {
                "source": "manual_hit_service",
                "target_track": target.get("target_track", {}),
            },
            "confidence": 1.0,
            "publishable": True,
            "reject_reason": "",
            "wall_time": time.time(),
            "command": {
                "command_id": command_id,
                "source": str(cmd.get("source") or ""),
                "click_xy": cmd.get("click_xy"),
            },
        }
        self.events_fp.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        rec = {
            "schema_version": 1,
            "type": "manual_hit_confirm",
            "source": "manual_hit_service",
            "accepted": True,
            "publishable": True,
            "event_id": event_id,
            "command_id": command_id,
            "source_global_id_int": int(source_gid),
            "target_global_id_int": int(target_gid),
            "target_global_id": str(event.get("target_global_id")),
            "target_display_player_id": str(event.get("target_display_player_id")),
            "target_display_slot_id": event.get("target_display_slot_id"),
            "target_id_source": str(event.get("target_id_source")),
            "resolve_reason": str(target.get("resolve_reason") or ""),
            "resolve_distance_m": target.get("resolve_distance_m"),
            "command": cmd,
            "event": event,
            "wall_time": time.time(),
            "wall_time_us": _now_us(),
        }
        self.confirmed_fp.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.latest_result = rec
        self.latest_event = event
        self.stats["accepted"] += 1
        self.stats["publishable_events"] += 1
        return rec

    def _process_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        self.latest_command = dict(cmd)
        self.stats["input_commands"] += 1
        if not self.mode_enabled:
            return self._reject(cmd, "service_mode_disabled")
        if not _as_bool(self.control.get("enabled"), True):
            return self._reject(cmd, "manual_hit_disabled")
        if str(cmd.get("type") or "") not in {"manual_hit_command", "preview_manual_hit_command", ""}:
            return self._reject(cmd, "unsupported_command_type")
        target, reason = self._resolve_target(cmd)
        if target is None:
            return self._reject(cmd, reason or "target_unresolved")
        target_gid = _safe_int(target.get("target_global_id_int"), -1)
        if target_gid < 0:
            return self._reject(cmd, "target_id_invalid", {"target": target})
        if target_gid == _safe_gid_int(self.control.get("source_global_id"), int(self.args.source_global_id)):
            return self._reject(cmd, "target_is_virtual_source", {"target_global_id_int": target_gid})
        track = target.get("target_track", {}) if isinstance(target.get("target_track"), dict) else {}
        age_ms = _safe_float(track.get("age_ms"), -1.0)
        max_age = max(0.0, float(self.control.get("target_max_age_ms", self.args.target_max_age_ms)))
        if age_ms >= 0 and age_ms > max_age:
            return self._reject(cmd, "target_track_stale", {"target_age_ms": age_ms, "target_max_age_ms": max_age})
        return self._accept(cmd, target)

    def _publish_status(self) -> None:
        now = _now_ms()
        command_age = -1.0
        if self.latest_command:
            ts = _safe_float(self.latest_command.get("wall_time_us"), 0.0)
            if ts > 0:
                command_age = max(0.0, (_now_us() - ts) / 1000.0)
        latest = {
            "schema_version": 1,
            "msg_type": "manual_hit_latest",
            "mode": str(self.args.mode),
            "timestamp_ms": now,
            "latest_result": dict(self.latest_result),
            "latest_event": dict(self.latest_event),
        }
        status = {
            "schema_version": 1,
            "source": "manual_hit_service",
            "service": "manual_hit",
            "title_cn": "人工命中确认",
            "mode": str(self.args.mode),
            "enabled": bool(self.mode_enabled and _as_bool(self.control.get("enabled"), True)),
            "affects_overall": False,
            "pid": os.getpid(),
            "command_jsonl": str(self.command_path),
            "confirmed_jsonl": str(self.confirmed_path),
            "events_jsonl": str(self.events_path),
            "control_json": str(self.control_path),
            "global_bev_status_json": str(self.global_bev_status_path),
            "person_latest_json": str(self.person_latest_path),
            "source_global_id": _safe_gid_int(self.control.get("source_global_id"), int(self.args.source_global_id)),
            "dedup_ms": float(self.control.get("dedup_ms", self.args.dedup_ms)),
            "target_max_age_ms": float(self.control.get("target_max_age_ms", self.args.target_max_age_ms)),
            "target_match_max_dist_m": float(self.control.get("target_match_max_dist_m", self.args.target_match_max_dist_m)),
            "input_age_ms": command_age,
            "output_age_ms": _safe_float(self.latest_result.get("wall_time_us"), 0.0),
            "control_seq": int(self.control_seq),
            "last_error": str(self.last_error),
            "tail_error": str(self.command_tailer.error),
            "stats": dict(self.stats),
            "latest_result": dict(self.latest_result),
            "timestamp_ms": now,
        }
        if status["output_age_ms"] > 0:
            status["output_age_ms"] = max(0.0, (_now_us() - float(status["output_age_ms"])) / 1000.0)
        else:
            status["output_age_ms"] = -1.0
        _atomic_write_json(self.latest_path, latest)
        _atomic_write_json(self.status_path, status)

    def run(self) -> None:
        print(
            f"[manual_hit] start mode={self.args.mode} commands={self.command_path} events={self.events_path}",
            flush=True,
        )
        next_status = 0.0
        while True:
            self._poll_control()
            commands = self.command_tailer.read_new(max_lines=int(self.args.max_lines_per_poll))
            for cmd in commands:
                try:
                    self._process_command(cmd)
                    self.last_error = ""
                except Exception as exc:
                    self.stats["process_errors"] += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._reject(cmd, "process_exception", {"error": self.last_error})
            now = time.time()
            if now >= next_status:
                self._publish_status()
                next_status = now + max(0.1, float(self.args.status_interval_sec))
            time.sleep(max(0.01, float(self.args.interval_sec)))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Publish preview-click manual hit commands")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--mode", choices=["disabled", "off", "shadow", "publish", "enforce"], default="publish")
    ap.add_argument("--command-jsonl", default="manual_hit_commands.jsonl")
    ap.add_argument("--confirmed-jsonl", default="manual_hit_confirmed.jsonl")
    ap.add_argument("--events-jsonl", default="damage_attribution_events.jsonl")
    ap.add_argument("--latest-json", default="manual_hit_latest.json")
    ap.add_argument("--status-json", default="manual_hit_status.json")
    ap.add_argument("--control-json", default="manual_hit_control.json")
    ap.add_argument("--global-bev-status-json", default="global_bev_status.json")
    ap.add_argument("--person-latest-json", default="global_person_latest.json")
    ap.add_argument("--source-global-id", type=int, default=99)
    ap.add_argument("--dedup-ms", type=float, default=800.0)
    ap.add_argument("--target-max-age-ms", type=float, default=1200.0)
    ap.add_argument("--target-match-max-dist-m", type=float, default=1.8)
    ap.add_argument("--interval-sec", type=float, default=0.04)
    ap.add_argument("--status-interval-sec", type=float, default=0.5)
    ap.add_argument("--max-lines-per-poll", type=int, default=512)
    ap.add_argument("--read-existing", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    ManualHitService(args).run()


if __name__ == "__main__":
    main()
