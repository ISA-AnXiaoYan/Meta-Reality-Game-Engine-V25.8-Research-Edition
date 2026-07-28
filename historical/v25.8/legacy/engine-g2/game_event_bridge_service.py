#!/usr/bin/env python3
"""Bridge runtime global IDs and hit events to the Windows game manager.

The Windows side keeps player identity, HP, damage, stats, and watches. This
process only sends current runtime global IDs (presence) and publishable hit
relations from damage_attribution_service.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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


class TcpJsonLineClient:
    def __init__(self, host: str, port: int, timeout_sec: float, reconnect_sec: float) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.reconnect_sec = float(reconnect_sec)
        self.sock: Optional[socket.socket] = None
        self.next_connect_t = 0.0
        self.last_error = ""
        self.connected_at = 0.0

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.connected_at = 0.0

    def connect_if_needed(self) -> bool:
        if self.sock is not None:
            return True
        now = time.time()
        if now < self.next_connect_t:
            return False
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            self.connected_at = now
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.next_connect_t = now + max(0.2, self.reconnect_sec)
            self.close()
            return False

    def send(self, payload: Dict[str, Any]) -> bool:
        if not self.connect_if_needed() or self.sock is None:
            return False
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self.sock.sendall(data)
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.next_connect_t = time.time() + max(0.2, self.reconnect_sec)
            self.close()
            return False


class GameEventBridgeService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.sync_root = Path(args.sync_root).resolve()
        self.person_latest_path = _resolve_path(self.sync_root, self.project_root, args.person_latest_json)
        self.global_bev_status_path = _resolve_path(self.sync_root, self.project_root, args.global_bev_status_json)
        self.events_path = _resolve_path(self.sync_root, self.project_root, args.events_jsonl)
        self.control_path = _resolve_path(self.sync_root, self.project_root, args.control_json)
        self.latest_path = _resolve_path(self.sync_root, self.project_root, args.latest_json)
        self.status_path = _resolve_path(self.sync_root, self.project_root, args.status_json)
        self.sent_jsonl_path = _resolve_path(self.sync_root, self.project_root, args.sent_jsonl)
        self.event_tailer = JsonlTailer(self.events_path, bool(args.read_existing))
        self.client = TcpJsonLineClient(args.windows_host, int(args.windows_port), float(args.tcp_timeout_sec), float(args.reconnect_sec))
        self.stats = Counter()
        self.sent_seen: Dict[str, float] = {}
        self.latest_payload: Dict[str, Any] = {}
        self.hit_source_mode = self._normalize_hit_source_mode(str(getattr(args, "hit_source_mode", "all") or "all"))
        self.control_mtime_ns = 0
        self.control_error = ""
        try:
            self.control_mtime_ns = int(self.control_path.stat().st_mtime_ns)
        except Exception:
            self.control_mtime_ns = 0
        self.sent_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.sent_fp = self.sent_jsonl_path.open("a", encoding="utf-8", buffering=1)

    @property
    def mode_enabled(self) -> bool:
        return str(self.args.mode) not in {"disabled", "off"}

    @property
    def send_hits_enabled(self) -> bool:
        return str(self.args.mode) in {"publish", "enforce"}

    @property
    def send_presence_enabled(self) -> bool:
        if str(self.args.mode) in {"publish", "enforce"}:
            return True
        return bool(getattr(self.args, "send_presence_in_shadow", False)) and str(self.args.mode) in {"shadow", "shadow_no_publish"}

    def _normalize_hit_source_mode(self, value: str) -> str:
        mode = str(value or "all").strip().lower()
        aliases = {
            "manual": "manual_only",
            "manual_hit": "manual_only",
            "manual_click": "manual_only",
            "manual_click_only": "manual_only",
            "auto": "auto_only",
            "automatic": "auto_only",
        }
        mode = aliases.get(mode, mode)
        return mode if mode in {"all", "manual_only", "auto_only"} else "all"

    def _poll_control(self) -> None:
        try:
            st = self.control_path.stat()
        except FileNotFoundError:
            self.control_error = ""
            return
        except Exception as exc:
            self.control_error = f"{type(exc).__name__}: {exc}"
            return
        if int(st.st_mtime_ns) <= int(self.control_mtime_ns):
            return
        try:
            obj = json.loads(self.control_path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict):
                if "hit_source_mode" in obj:
                    self.hit_source_mode = self._normalize_hit_source_mode(str(obj.get("hit_source_mode") or "all"))
                elif "manual_hit_only" in obj:
                    self.hit_source_mode = "manual_only" if bool(obj.get("manual_hit_only")) else "all"
            self.control_mtime_ns = int(st.st_mtime_ns)
            self.control_error = ""
        except Exception as exc:
            self.control_error = f"{type(exc).__name__}: {exc}"

    def _is_manual_hit_event(self, event: Dict[str, Any]) -> bool:
        if bool(event.get("manual", False)):
            return True
        source = str(event.get("source") or "").strip().lower()
        attribution = str(event.get("attribution_source") or "").strip().lower()
        authority = str(event.get("manual_authority") or "").strip()
        return bool(authority) or source in {"manual_hit_service", "manual_preview_click"} or attribution == "manual_preview_click"

    def _hit_source_allowed(self, event: Dict[str, Any]) -> bool:
        mode = self._normalize_hit_source_mode(self.hit_source_mode)
        is_manual = self._is_manual_hit_event(event)
        if mode == "manual_only" and not is_manual:
            self.stats["suppress_non_manual_hit_by_source_mode"] += 1
            return False
        if mode == "auto_only" and is_manual:
            self.stats["suppress_manual_hit_by_source_mode"] += 1
            return False
        return True

    def _ensure_virtual_presence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        virtual_id = _safe_int(getattr(self.args, "virtual_presence_global_id", -1), -1)
        if virtual_id < 0:
            return payload
        payload = dict(payload)
        player_id = str(getattr(self.args, "virtual_presence_player_id", "") or f"P{virtual_id:02d}")

        raw_ids = payload.get("alive_yolo_ids", [])
        ids: List[int] = []
        seen_ids = set()
        if isinstance(raw_ids, list):
            for value in raw_ids:
                gid = _safe_int(value, -1)
                if gid >= 0 and gid not in seen_ids:
                    ids.append(gid)
                    seen_ids.add(gid)
        if virtual_id not in seen_ids:
            ids.append(virtual_id)
            seen_ids.add(virtual_id)
            self.stats["virtual_presence_added_to_alive_ids"] += 1
        payload["alive_yolo_ids"] = sorted(ids)

        def item_global_id(item: Dict[str, Any]) -> int:
            gid = _safe_int(item.get("global_id_int"), -1)
            if gid >= 0:
                return gid
            label = str(item.get("global_id") or "").strip().upper()
            if label.startswith("G"):
                return _safe_int(label[1:], -1)
            return _safe_int(label, -1)

        raw_full = payload.get("global_ids", [])
        full: List[Dict[str, Any]] = []
        if isinstance(raw_full, list):
            for item in raw_full:
                if not isinstance(item, dict):
                    continue
                if item_global_id(item) == virtual_id:
                    self.stats["virtual_presence_replaced_existing"] += 1
                    continue
                full.append(dict(item))
        full.append({
            "global_id": _gid_str(virtual_id),
            "global_id_int": int(virtual_id),
            "global_player_id": player_id,
            "id_source": "virtual_global_id",
            "source_global_id": _gid_str(virtual_id),
            "source_global_id_int": int(virtual_id),
            "source_global_player_id": player_id,
            "state": "virtual_present",
            "age_ms": 0.0,
            "confidence": 1.0,
            "source_cameras": [],
            "virtual": True,
            "role": "damage_source",
        })
        full.sort(key=lambda x: int(_safe_int(x.get("global_id_int"), 0)))
        payload["global_ids"] = full
        payload["virtual_presence_global_id"] = int(virtual_id)
        payload["virtual_presence_player_id"] = player_id
        payload["virtual_presence_present"] = True
        self.stats["virtual_presence_injected"] += 1
        return payload

    def _load_presence(self) -> Dict[str, Any]:
        if str(getattr(self.args, "id_source", "global_runtime") or "global_runtime") == "global_bev_display":
            display = self._load_display_presence()
            if display is not None:
                return display
            self.stats["presence_display_not_ready"] += 1
            return self._ensure_virtual_presence({
                "type": "presence",
                "alive_yolo_ids": [],
                "global_ids": [],
                "id_semantics": "global_bev_display_id",
                "source_id_semantics": "global_runtime_id",
                "id_source_state": "global_bev_display_not_ready",
                "ts_ms": _now_ms(),
            })
        obj = _read_json(self.person_latest_path) or {}
        tracks = obj.get("tracks", [])
        if not isinstance(tracks, list):
            tracks = []
        ids: List[int] = []
        full: List[Dict[str, Any]] = []
        max_age = float(self.args.presence_max_age_ms)
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            gid = _safe_int(tr.get("global_id"), -1)
            if gid < 0:
                continue
            age = _safe_float(tr.get("age_ms"), 0.0)
            if max_age >= 0 and age > max_age:
                continue
            ids.append(gid)
            full.append({
                "global_id": _gid_str(gid),
                "global_id_int": int(gid),
                "global_player_id": str(tr.get("global_player_id") or f"P{gid:02d}"),
                "state": str(tr.get("state", "")),
                "age_ms": round(float(age), 2),
                "confidence": _safe_float(tr.get("confidence"), 0.0),
                "source_cameras": list(tr.get("source_cameras", [])) if isinstance(tr.get("source_cameras"), list) else [],
            })
        ids = sorted(set(ids))
        full.sort(key=lambda x: int(x.get("global_id_int", 0)))
        return self._ensure_virtual_presence({
            "type": "presence",
            "alive_yolo_ids": ids,
            "global_ids": full,
            "id_semantics": "global_runtime_id",
            "ts_ms": _now_ms(),
        })

    def _load_display_presence(self) -> Optional[Dict[str, Any]]:
        obj = _read_json(self.global_bev_status_path) or {}
        gp = obj.get("global_person", obj.get("global_person_overlay", {}))
        if not isinstance(gp, dict):
            self.stats["presence_display_missing_global_person_status"] += 1
            return None
        tracks = gp.get("display_tracks", [])
        if not isinstance(tracks, list):
            self.stats["presence_display_missing_tracks"] += 1
            return None
        max_age = float(getattr(self.args, "global_bev_display_max_age_ms", 1200.0))
        ids: List[int] = []
        full: List[Dict[str, Any]] = []
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            slot_id = _safe_int(tr.get("display_slot_id"), -1)
            if slot_id < 0:
                label = str(tr.get("display_player_id") or "").strip().upper()
                if label.startswith("P"):
                    slot_id = _safe_int(label[1:], -1)
            if slot_id < 0:
                continue
            age = _safe_float(tr.get("age_ms"), 0.0)
            if max_age >= 0.0 and age > max_age:
                continue
            source_gid = _safe_int(tr.get("source_global_id"), -1)
            source_label = str(tr.get("source_global_player_id") or (f"P{source_gid:02d}" if source_gid >= 0 else ""))
            display_label = str(tr.get("display_player_id") or f"P{slot_id:02d}")
            ids.append(slot_id)
            full.append({
                "global_id": _gid_str(slot_id),
                "global_id_int": int(slot_id),
                "global_player_id": display_label,
                "id_source": "global_bev_display",
                "source_global_id": _gid_str(source_gid) if source_gid >= 0 else "",
                "source_global_id_int": int(source_gid) if source_gid >= 0 else -1,
                "source_global_player_id": source_label,
                "state": str(tr.get("state", "")),
                "age_ms": round(float(age), 2),
                "confidence": _safe_float(tr.get("confidence"), 0.0),
                "source_cameras": list(tr.get("source_cameras", [])) if isinstance(tr.get("source_cameras"), list) else [],
            })
        ids = sorted(set(ids))
        full.sort(key=lambda x: int(x.get("global_id_int", 0)))
        payload = self._ensure_virtual_presence({
            "type": "presence",
            "alive_yolo_ids": ids,
            "global_ids": full,
            "id_semantics": "global_bev_display_id",
            "source_id_semantics": "global_runtime_id",
            "ts_ms": _now_ms(),
        })
        if not payload.get("alive_yolo_ids"):
            self.stats["presence_display_empty"] += 1
            return None
        self.stats["presence_display_used"] += 1
        return payload

    def _hit_payload(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not bool(event.get("publishable", False)):
            self.stats["skip_not_publishable"] += 1
            return None
        if not self._hit_source_allowed(event):
            return None
        source_id = _safe_int(event.get("source_global_id_int"), -1)
        target_id = _safe_int(event.get("target_global_id_int"), -1)
        forced_source_id = _safe_int(getattr(self.args, "force_hit_source_global_id", -1), -1)
        if forced_source_id >= 0:
            if source_id != forced_source_id:
                self.stats["force_hit_source_global_id_override"] += 1
            source_id = int(forced_source_id)
        if source_id < 0 or target_id < 0:
            self.stats["skip_missing_source_or_target"] += 1
            return None
        reserved_virtual_id = _safe_int(getattr(self.args, "virtual_presence_global_id", -1), -1)
        if reserved_virtual_id >= 0 and target_id == reserved_virtual_id:
            self.stats["skip_target_reserved_virtual_id"] += 1
            return None
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            self.stats["skip_missing_event_id"] += 1
            return None
        return {
            "type": "hit",
            "source_yolo_id": int(source_id),
            "target_yolo_id": int(target_id),
            "source_global_id": _gid_str(source_id),
            "target_global_id": str(event.get("target_global_id") or _gid_str(target_id)),
            "target_id_source": str(event.get("target_id_source") or ""),
            "target_source_global_id": str(event.get("target_source_global_id") or ""),
            "target_source_global_id_int": _safe_int(event.get("target_source_global_id_int"), -1),
            "target_display_slot_id": _safe_int(event.get("target_display_slot_id"), -1),
            "target_display_player_id": str(event.get("target_display_player_id") or ""),
            "target_identity_evidence": dict(event.get("target_identity_evidence", {})) if isinstance(event.get("target_identity_evidence"), dict) else {},
            "target_display_evidence": dict(event.get("target_display_evidence", {})) if isinstance(event.get("target_display_evidence"), dict) else {},
            "event_id": event_id,
            "camera": str(event.get("camera") or "global"),
            "id_semantics": str(event.get("id_semantics") or "global_runtime_id"),
            "confidence": _safe_float(event.get("confidence"), 0.0),
            "source_override": bool(forced_source_id >= 0),
            "event_source": str(event.get("source") or ""),
            "manual": bool(event.get("manual", False)),
            "manual_authority": str(event.get("manual_authority") or ""),
            "attribution_source": str(event.get("attribution_source") or ""),
        }

    def _dedup_hit(self, payload: Dict[str, Any]) -> bool:
        camera = str(payload.get("camera") or "")
        event_id = str(payload.get("event_id") or "")
        key = f"{camera}:{event_id}"
        now = time.time()
        ttl = max(1.0, float(self.args.event_id_ttl_sec))
        expired = [k for k, t in self.sent_seen.items() if now - t > ttl]
        for k in expired[:1024]:
            self.sent_seen.pop(k, None)
        if key in self.sent_seen:
            self.stats["dedup_hit"] += 1
            return True
        self.sent_seen[key] = now
        return False

    def _write_sent_log(self, payload: Dict[str, Any], sent: bool, reason: str) -> None:
        rec = {
            "wall_time": time.time(),
            "mode": str(self.args.mode),
            "sent": bool(sent),
            "reason": str(reason),
            "payload": payload,
        }
        self.sent_fp.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.latest_payload = rec

    def _send_or_shadow(self, payload: Dict[str, Any], allow_send: bool, reason: str) -> None:
        sent = False
        if allow_send:
            sent = self.client.send(payload)
            if sent:
                self.stats[f"sent_{payload.get('type', 'unknown')}"] += 1
            else:
                self.stats[f"send_failed_{payload.get('type', 'unknown')}"] += 1
        else:
            self.stats[f"shadow_{payload.get('type', 'unknown')}"] += 1
        self._write_sent_log(payload, sent, reason if sent else ("shadow" if not allow_send else "send_failed"))

    def _publish_status(self) -> None:
        latest = {
            "schema_version": 1,
            "msg_type": "game_event_bridge_latest",
            "mode": str(self.args.mode),
            "timestamp_ms": _now_ms(),
            "latest_payload": dict(self.latest_payload),
        }
        status = {
            "schema_version": 1,
            "source": "game_event_bridge_service",
            "mode": str(self.args.mode),
            "windows_host": str(self.args.windows_host),
            "windows_port": int(self.args.windows_port),
            "send_presence_enabled": bool(self.send_presence_enabled),
            "send_hits_enabled": bool(self.send_hits_enabled),
            "id_source": str(getattr(self.args, "id_source", "global_runtime")),
            "hit_source_mode": str(self.hit_source_mode),
            "control_json": str(self.control_path),
            "control_error": str(self.control_error),
            "force_hit_source_global_id": int(getattr(self.args, "force_hit_source_global_id", -1)),
            "virtual_presence_global_id": int(getattr(self.args, "virtual_presence_global_id", -1)),
            "virtual_presence_player_id": str(getattr(self.args, "virtual_presence_player_id", "")),
            "person_latest_json": str(self.person_latest_path),
            "global_bev_status_json": str(self.global_bev_status_path),
            "events_jsonl": str(self.events_path),
            "sent_jsonl": str(self.sent_jsonl_path),
            "client_connected": bool(self.client.sock is not None),
            "client_last_error": str(self.client.last_error),
            "stats": dict(self.stats),
            "event_tail_error": str(self.event_tailer.error),
            "timestamp_ms": _now_ms(),
        }
        _atomic_write_json(self.latest_path, latest)
        _atomic_write_json(self.status_path, status)

    def run(self) -> None:
        print(
            f"[game_event_bridge] start mode={self.args.mode} target={self.args.windows_host}:{self.args.windows_port}",
            flush=True,
        )
        next_presence = 0.0
        next_status = 0.0
        while True:
            self._poll_control()
            if not self.mode_enabled:
                self._publish_status()
                time.sleep(max(0.1, float(self.args.interval_sec)))
                continue
            now = time.time()
            if now >= next_presence:
                presence = self._load_presence()
                self._send_or_shadow(presence, self.send_presence_enabled, "presence")
                hz = max(0.1, float(self.args.presence_hz))
                next_presence = now + 1.0 / hz
            for event in self.event_tailer.read_new(max_lines=int(self.args.max_lines_per_poll)):
                payload = self._hit_payload(event)
                if payload is None:
                    continue
                if self._dedup_hit(payload):
                    continue
                self._send_or_shadow(payload, self.send_hits_enabled, "hit")
            if now >= next_status:
                self._publish_status()
                next_status = now + max(0.1, float(self.args.status_interval_sec))
            time.sleep(max(0.01, float(self.args.interval_sec)))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Bridge runtime global IDs and hit events to Windows game manager")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--mode", choices=["disabled", "shadow_no_publish", "shadow", "publish", "enforce"], default="disabled")
    ap.add_argument("--person-latest-json", default="global_person_latest.json")
    ap.add_argument("--global-bev-status-json", default="global_bev_status.json")
    ap.add_argument("--id-source", choices=["global_runtime", "global_bev_display"], default="global_runtime")
    ap.add_argument("--global-bev-display-max-age-ms", type=float, default=1200.0)
    ap.add_argument("--events-jsonl", default="damage_attribution_events.jsonl")
    ap.add_argument("--control-json", default="game_event_bridge_control.json")
    ap.add_argument("--hit-source-mode", choices=["all", "manual_only", "auto_only"], default="all")
    ap.add_argument("--latest-json", default="game_event_bridge_latest.json")
    ap.add_argument("--status-json", default="game_event_bridge_status.json")
    ap.add_argument("--sent-jsonl", default="game_event_bridge_sent.jsonl")
    ap.add_argument("--windows-host", default="192.168.31.76")
    ap.add_argument("--windows-port", type=int, default=7002)
    ap.add_argument("--presence-hz", type=float, default=5.0)
    ap.add_argument("--presence-max-age-ms", type=float, default=800.0)
    ap.add_argument("--send-presence-in-shadow", action="store_true")
    ap.add_argument("--interval-sec", type=float, default=0.1)
    ap.add_argument("--status-interval-sec", type=float, default=1.0)
    ap.add_argument("--reconnect-sec", type=float, default=3.0)
    ap.add_argument("--tcp-timeout-sec", type=float, default=1.5)
    ap.add_argument("--event-id-ttl-sec", type=float, default=60.0)
    ap.add_argument("--max-lines-per-poll", type=int, default=1000)
    ap.add_argument("--force-hit-source-global-id", type=int, default=-1)
    ap.add_argument("--virtual-presence-global-id", type=int, default=-1)
    ap.add_argument("--virtual-presence-player-id", default="")
    ap.add_argument("--read-existing", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    GameEventBridgeService(args).run()


if __name__ == "__main__":
    main()
