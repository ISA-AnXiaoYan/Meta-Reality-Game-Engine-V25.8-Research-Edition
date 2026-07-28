#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CAMERA_IDS = tuple("ABCDEFGH")


def now_wall_us() -> int:
    return int(time.time() * 1_000_000)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text)
        if not isinstance(data, dict):
            data = {"value": data}
        st = path.stat()
        data["_path"] = str(path)
        data["_mtime"] = st.st_mtime
        data["_age_ms"] = max(0.0, (time.time() - st.st_mtime) * 1000.0)
        return data
    except Exception as exc:
        return {"_path": str(path), "_error": repr(exc)}


def first_dict(obj: Dict[str, Any], *keys: str) -> Tuple[Dict[str, Any], str]:
    if not isinstance(obj, dict):
        return {}, ""
    for key in keys:
        val = obj.get(key)
        if isinstance(val, dict):
            return val, key
    return {}, ""


def has_path(obj: Dict[str, Any], path: str) -> bool:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur.get(part)
    return True


def collect_status_key_audit(bev: Dict[str, Any], overview: Dict[str, Any],
                             system: Dict[str, Any], hit: Dict[str, Any],
                             bev_control: Dict[str, Any]) -> Dict[str, Any]:
    categories = system.get("categories", {}) if isinstance(system.get("categories"), dict) else {}
    gb = categories.get("global_bev", {}) if isinstance(categories.get("global_bev"), dict) else {}
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, source: str = "", note: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "source": source, "note": note})

    bev_layer, bev_layer_src = first_dict(bev, "event_layer", "event_layer_overlay")
    _, bev_bullet_src = first_dict(bev, "event_bullet", "event_bullet_overlay")
    _, bev_person_src = first_dict(bev, "global_person", "global_person_overlay")
    _, cat_layer_src = first_dict(gb, "event_layer", "event_layer_overlay")
    _, cat_bullet_src = first_dict(gb, "event_bullet", "event_bullet_overlay")
    _, cat_person_src = first_dict(gb, "global_person", "global_person_overlay")

    add("global_bev_status.event_layer", bool(bev_layer_src), bev_layer_src, "preferred short key, legacy event_layer_overlay accepted")
    add("global_bev_status.event_bullet", bool(bev_bullet_src), bev_bullet_src, "preferred short key, legacy event_bullet_overlay accepted")
    add("global_bev_status.global_person", bool(bev_person_src), bev_person_src, "preferred short key, legacy global_person_overlay accepted")
    add("global_bev_status.readback", isinstance(bev.get("readback"), dict), "readback" if isinstance(bev.get("readback"), dict) else "")
    add("global_bev_status.control", isinstance(bev.get("control"), dict), "control" if isinstance(bev.get("control"), dict) else "")
    add("global_bev_status.alignment", isinstance(bev.get("alignment"), dict), "alignment" if isinstance(bev.get("alignment"), dict) else "")
    add("global_bev_status.mvs_texture_upload", isinstance(bev.get("mvs_texture_upload"), dict), "mvs_texture_upload" if isinstance(bev.get("mvs_texture_upload"), dict) else "")
    add("global_bev_status.event_layer.display_budget_ms", "display_budget_ms" in bev_layer, "display_budget_ms" if "display_budget_ms" in bev_layer else "")
    add("global_bev_status.event_layer.display_budget_overrun_rate", "display_budget_overrun_rate" in bev_layer, "display_budget_overrun_rate" if "display_budget_overrun_rate" in bev_layer else "")
    add("global_bev_status.event_layer.display_budget_order", "display_budget_order" in bev_layer, "display_budget_order" if "display_budget_order" in bev_layer else "")
    presentation_worker = bev_layer.get("presentation_worker", {}) if isinstance(bev_layer.get("presentation_worker"), dict) else {}
    add("global_bev_status.event_layer.presentation_worker", True, "presentation_worker" if presentation_worker else "", "optional before V24-E4; required in V24-E4 validation")
    if presentation_worker:
        add("global_bev_status.event_layer.presentation_worker.mode", "mode" in presentation_worker, "mode" if "mode" in presentation_worker else "")
        add("global_bev_status.event_layer.presentation_worker.state", "state" in presentation_worker, "state" if "state" in presentation_worker else "")
    mvs_tex = bev.get("mvs_texture_upload", {}) if isinstance(bev.get("mvs_texture_upload"), dict) else {}
    add("global_bev_status.mvs_texture_upload.budget_overrun_rate", "budget_overrun_rate" in mvs_tex, "budget_overrun_rate" if "budget_overrun_rate" in mvs_tex else "")

    for key in ("mvs", "event", "global_yolo", "global_person_id", "global_bev", "hit_judge", "bullet_reprocess", "raw_recording", "person_id_dataset", "source_mode_replay", "preview"):
        add(f"system_status.categories.{key}", isinstance(categories.get(key), dict), key if isinstance(categories.get(key), dict) else "")
    source_mode_present = isinstance(categories.get("source_mode_replay"), dict)
    if source_mode_present:
        source_mode_cat = categories.get("source_mode_replay", {})
        for key in (
            "event_source_mode_effective",
            "mvs_source_mode_effective",
            "source_mode_effective_exists",
            "source_mode_effective_path",
            "v25_resolved_manifest_exists",
            "v25_resolved_manifest_path",
            "event_file_fifo_broker_enabled",
            "mvs_file_frame_broker_enabled",
            "usable_for_full_replay_startup",
            "usable_for_full_replay_sync",
            "v25_replay_control_exists",
            "v25_replay_status_exists",
        ):
            add(
                f"system_status.categories.source_mode_replay.{key}",
                key in source_mode_cat,
                key if key in source_mode_cat else "",
            )
    exp_present = isinstance(categories.get("experiment_control"), dict)
    add(
        "system_status.categories.experiment_control",
        True,
        "experiment_control" if exp_present else "",
        "optional during rollout; authoritative experiment state is also in sync_ipc/experiment_status_latest.json",
    )
    if exp_present:
        exp_cat = categories.get("experiment_control", {})
        for key in (
            "experiment_id",
            "stage",
            "config_hash",
            "state",
            "final_hit_authority",
            "human_noise_filter_mode",
            "human_noise_track_start_bbox_expand_px",
            "human_noise_track_start_bbox_policy",
            "global_person_target",
            "global_person_filter_backend",
        ):
            add(
                f"system_status.categories.experiment_control.{key}",
                key in exp_cat,
                key if key in exp_cat else "",
            )
    out_hit_present = isinstance(categories.get("outbound_hit_events"), dict)
    add(
        "system_status.categories.outbound_hit_events",
        True,
        "outbound_hit_events" if out_hit_present else "",
        "optional diagnostics window for game_event_bridge_sent.jsonl during rollout",
    )
    if out_hit_present:
        out_hit = categories.get("outbound_hit_events", {})
        for key in (
            "state",
            "mode",
            "send_hits_enabled",
            "client_connected",
            "sent_jsonl",
            "recent_hit_count",
            "recent_sent_count",
            "recent_send_failed_count",
            "total_sent_hit",
            "latest_hit_age_ms",
        ):
            add(
                f"system_status.categories.outbound_hit_events.{key}",
                key in out_hit,
                key if key in out_hit else "",
            )

    add("system_status.categories.global_bev.event_layer", bool(cat_layer_src), cat_layer_src)
    add("system_status.categories.global_bev.event_bullet", bool(cat_bullet_src), cat_bullet_src)
    add("system_status.categories.global_bev.global_person", bool(cat_person_src), cat_person_src)
    add("system_status.categories.global_bev.pbo_ready", has_path(gb, "pbo_ready"), "pbo_ready" if has_path(gb, "pbo_ready") else "")
    add(
        "global_bev_control.event_layer_enable",
        "event_layer_enable" in bev_control or "show_event_layer" in bev_control,
        "event_layer_enable" if "event_layer_enable" in bev_control else ("show_event_layer" if "show_event_layer" in bev_control else ""),
        "preferred event_layer_enable, legacy show_event_layer accepted",
    )
    add("global_bev_control.event_layer_dilate_px", "event_layer_dilate_px" in bev_control, "event_layer_dilate_px" if "event_layer_dilate_px" in bev_control else "")
    add("event_overlay_overview_status.control", isinstance(overview.get("control"), dict), "control" if isinstance(overview.get("control"), dict) else "")
    add("hit_judge_status.hit_detection_enabled", "hit_detection_enabled" in hit or "enabled" in hit, "hit_detection_enabled" if "hit_detection_enabled" in hit else ("enabled" if "enabled" in hit else ""))
    add("hit_judge_status.strong_gate_enabled", "strong_gate_enabled" in hit, "strong_gate_enabled" if "strong_gate_enabled" in hit else "")
    add("hit_judge_status.strong_reasoning_enabled", "strong_reasoning_enabled" in hit, "strong_reasoning_enabled" if "strong_reasoning_enabled" in hit else "")
    add("hit_judge_status.final_hit_authority", "final_hit_authority" in hit, "final_hit_authority" if "final_hit_authority" in hit else "")
    add("hit_judge_status.pseudo_hit_total", "pseudo_hit_total" in hit, "pseudo_hit_total" if "pseudo_hit_total" in hit else "")
    add("hit_judge_status.human_noise_filter", isinstance(hit.get("human_noise_filter"), dict), "human_noise_filter" if isinstance(hit.get("human_noise_filter"), dict) else "")
    add("hit_judge_status.pseudo_hit_human_noise_suppressed", "pseudo_hit_human_noise_suppressed" in hit, "pseudo_hit_human_noise_suppressed" if "pseudo_hit_human_noise_suppressed" in hit else "")

    missing = [c["name"] for c in checks if not c["ok"]]
    return {
        "ok": not missing,
        "missing": missing,
        "checks": checks,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as fp:
        fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_hash(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def mask_stats(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted((root / "sync_ipc").glob("event_human_mask_stats_*.jsonl")):
        try:
            st = path.stat()
            out.append({
                "name": path.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "age_ms": max(0.0, (time.time() - st.st_mtime) * 1000.0),
            })
        except OSError:
            continue
    return out


def parse_mask_list(path: Path) -> Dict[str, Tuple[int, float]]:
    out: Dict[str, Tuple[int, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            out[parts[0]] = (int(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return out


def read_text(path: Path, max_chars: int = 8000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_chars:
            return data[-max_chars:]
        return data
    except Exception as exc:
        return f"<read_error {exc!r}>"


def read_jsonl_last(path: Path, max_bytes: int = 1_000_000) -> Dict[str, Any]:
    if not path.exists():
        return {"_path": str(path), "_missing": True}
    try:
        st = path.stat()
        with path.open("rb") as fp:
            size = st.st_size
            fp.seek(max(0, size - max_bytes))
            data = fp.read()
        text = data.decode("utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                item = {"value": item}
            item["_path"] = str(path)
            item["_mtime"] = st.st_mtime
            item["_age_ms"] = max(0.0, (time.time() - st.st_mtime) * 1000.0)
            item["_size"] = st.st_size
            return item
        return {"_path": str(path), "_error": "no valid json object in tail", "_size": st.st_size}
    except Exception as exc:
        return {"_path": str(path), "_error": repr(exc)}


def summarize_event_broker_stats(run_dir: Path) -> Dict[str, Any]:
    candidates = [
        run_dir / "status" / "event_native_hal_broker_stats.jsonl",
        run_dir / "status" / "event_file_fifo_broker_stats.jsonl",
    ]
    for path in sorted((run_dir / "status").glob("event_native_hal_broker_stats*.jsonl")):
        if path not in candidates:
            candidates.append(path)
    for path in sorted((run_dir / "status").glob("event_file_fifo_broker_stats*.jsonl")):
        if path not in candidates:
            candidates.append(path)

    selected: Dict[str, Any] = {}
    for path in candidates:
        item = read_jsonl_last(path)
        if item.get("_missing"):
            continue
        if not selected or float(item.get("_mtime", 0.0) or 0.0) >= float(selected.get("_mtime", 0.0) or 0.0):
            selected = item
    if not selected:
        return {"available": False, "checked": [str(p) for p in candidates]}

    cameras = selected.get("cameras", [])
    compact_cameras: List[Dict[str, Any]] = []
    if isinstance(cameras, list):
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            raw_file = str(cam.get("raw_file") or "")
            compact_cameras.append({
                "alias": cam.get("alias"),
                "opened": cam.get("opened"),
                "started": cam.get("started"),
                "finished": cam.get("finished"),
                "has_ext_trigger": cam.get("has_ext_trigger"),
                "valid_for_replay_sync": cam.get("valid_for_replay_sync"),
                "events_seen": cam.get("events_seen"),
                "triggers_seen": cam.get("triggers_seen"),
                "frames_written": cam.get("frames_written"),
                "write_errors": cam.get("write_errors"),
                "exceptions": cam.get("exceptions"),
                "last_error": cam.get("last_error"),
                "median_ext_trigger_period_us": cam.get("median_ext_trigger_period_us"),
                "max_ext_trigger_gap_us": cam.get("max_ext_trigger_gap_us"),
                "raw_file_name": Path(raw_file).name if raw_file else None,
            })

    valid_count = sum(1 for cam in compact_cameras if bool(cam.get("valid_for_replay_sync")))
    opened_count = sum(1 for cam in compact_cameras if bool(cam.get("opened")))
    write_errors_total = sum(int(cam.get("write_errors") or 0) for cam in compact_cameras)
    exceptions_total = sum(int(cam.get("exceptions") or 0) for cam in compact_cameras)
    error_count = write_errors_total + exceptions_total
    return {
        "available": True,
        "path": selected.get("_path"),
        "schema": selected.get("schema"),
        "source_mode": selected.get("source_mode"),
        "sync_policy": selected.get("sync_policy"),
        "tick": selected.get("tick"),
        "published_wall_us": selected.get("published_wall_us"),
        "age_ms": selected.get("_age_ms"),
        "output_mode": selected.get("output_mode"),
        "replay_timing": selected.get("replay_timing"),
        "time_shift": selected.get("time_shift"),
        "slice_us": selected.get("slice_us"),
        "allow_missing_ext_trigger": selected.get("allow_missing_ext_trigger"),
        "cameras_requested": selected.get("cameras_requested"),
        "cameras_opened": selected.get("cameras_opened", opened_count),
        "cameras_finished": selected.get("cameras_finished"),
        "events_seen": selected.get("events_seen"),
        "triggers_seen": selected.get("triggers_seen"),
        "frames_written": selected.get("frames_written"),
        "errors": selected.get("errors", error_count),
        "write_errors_total": write_errors_total,
        "exceptions_total": exceptions_total,
        "valid_for_replay_sync_count": valid_count,
        "cameras": compact_cameras,
    }


def _requested_mvs_file_replay(source_mode_request: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(source_mode_request, dict) or not source_mode_request:
        return True
    requested = str(source_mode_request.get("mvs_source_mode_requested") or "").strip().lower()
    if requested in {"file_replay", "mvs_file_replay"}:
        return True
    extra = source_mode_request.get("launch_extra_args")
    if isinstance(extra, list):
        extra_text = " ".join(str(x) for x in extra)
    else:
        extra_text = str(extra or "")
    extra_text_l = extra_text.lower()
    return (
        "--mvs-source-mode file_replay" in extra_text_l
        or "--mvs-source-mode=file_replay" in extra_text_l
        or "--mvs-source-mode mvs_file_replay" in extra_text_l
        or "--mvs-source-mode=mvs_file_replay" in extra_text_l
    )


def summarize_mvs_file_broker_stats(
        run_dir: Path,
        source_mode_request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _requested_mvs_file_replay(source_mode_request):
        requested = ""
        if isinstance(source_mode_request, dict):
            requested = str(source_mode_request.get("mvs_source_mode_requested") or "")
        return {
            "available": False,
            "skipped": True,
            "skipped_reason": "mvs_source_mode_not_file_replay",
            "mvs_source_mode_requested": requested,
        }
    candidates = [
        run_dir / "status" / "mvs_file_frame_broker_stats.jsonl",
        run_dir / "status" / "mvs_file_frame_broker_status.json",
    ]
    for path in sorted((run_dir / "status").glob("mvs_file_frame_broker_stats*.jsonl")):
        if path not in candidates:
            candidates.append(path)

    selected: Dict[str, Any] = {}
    for path in candidates:
        item = read_jsonl_last(path) if path.suffix == ".jsonl" else read_json(path)
        if item.get("_missing") or item.get("_error"):
            continue
        if not selected or float(item.get("_mtime", 0.0) or 0.0) >= float(selected.get("_mtime", 0.0) or 0.0):
            selected = item
    if not selected:
        return {"available": False, "checked": [str(p) for p in candidates]}

    cameras = selected.get("cameras", [])
    compact_cameras: List[Dict[str, Any]] = []
    if isinstance(cameras, list):
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            compact_cameras.append({
                "camera": cam.get("camera"),
                "frames_published": cam.get("frames_published"),
                "finished": cam.get("finished"),
                "errors": cam.get("errors"),
                "last_error": cam.get("last_error"),
                "video_name": Path(str(cam.get("video_path") or "")).name if cam.get("video_path") else None,
                "index_name": Path(str(cam.get("index_path") or "")).name if cam.get("index_path") else None,
            })
    return {
        "available": True,
        "path": selected.get("_path"),
        "schema": selected.get("schema"),
        "source_mode": selected.get("source_mode"),
        "dataset_dir": selected.get("dataset_dir"),
        "state": selected.get("state"),
        "tick": selected.get("tick"),
        "age_ms": selected.get("_age_ms"),
        "published_wall_us": selected.get("published_wall_us"),
        "replay_timing": selected.get("replay_timing"),
        "raw_preview_enable": selected.get("raw_preview_enable"),
        "cameras_requested": selected.get("cameras_requested"),
        "total_frames_published": selected.get("total_frames_published"),
        "cameras_published": sum(1 for cam in compact_cameras if int(cam.get("frames_published") or 0) > 0),
        "error_count": sum(int(cam.get("errors") or 0) for cam in compact_cameras),
        "cameras": compact_cameras,
    }


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def collect_event_worker_sync_diagnostics(sync: Path, bev: Dict[str, Any]) -> Dict[str, Any]:
    alignment = bev.get("alignment", {}) if isinstance(bev.get("alignment"), dict) else {}
    bev_target = _first_present(bev.get("target_sync_id"), alignment.get("target_sync_id"))
    bev_target_i: Optional[int]
    if bev_target is None:
        bev_target_i = None
    else:
        bev_target_i = to_int(bev_target, 0)

    cams: Dict[str, Any] = {}
    mapped_values: List[int] = []
    direct_index_count = 0
    locked_count = 0
    trigger_edge_count = 0

    for cam in CAMERA_IDS:
        status = read_json(sync / f"event_worker_{cam}_status.json")
        event_sync = status.get("event_sync", {}) if isinstance(status.get("event_sync"), dict) else {}
        ext_trigger = status.get("ext_trigger", {}) if isinstance(status.get("ext_trigger"), dict) else {}

        mapped_raw = _first_present(
            ext_trigger.get("last_mapped_mvs_sync_index"),
            event_sync.get("last_mapped_mvs_sync_index"),
            event_sync.get("mapped_mvs_sync_index"),
        )
        mapped_i: Optional[int] = None if mapped_raw is None else to_int(mapped_raw, 0)
        if mapped_i is not None:
            mapped_values.append(mapped_i)

        lock_state = str(_first_present(
            event_sync.get("lock_state"),
            event_sync.get("event_sync_lock_state"),
            status.get("event_sync_lock_state"),
            "",
        ) or "")
        direct_index = bool(event_sync.get("direct_index_map", status.get("event_sync_direct_index_map", False)))
        if direct_index:
            direct_index_count += 1
        if lock_state and lock_state.upper() not in {"", "UNLOCKED", "WAITING", "NO_SYNC"}:
            locked_count += 1

        selected_edges = to_int(ext_trigger.get("selected_edge_count"), 0)
        all_edges = to_int(ext_trigger.get("all_edge_count"), 0)
        trigger_edge_count += selected_edges

        cams[cam] = {
            "status_path": status.get("_path"),
            "status_age_ms": status.get("_age_ms"),
            "state": status.get("state"),
            "active_reason": status.get("active_reason"),
            "ts_us": status.get("ts_us"),
            "event_sync_lock_state": lock_state,
            "event_sync_match_mode": _first_present(
                event_sync.get("match_mode"),
                event_sync.get("event_sync_match_mode"),
                status.get("event_sync_match_mode"),
            ),
            "direct_index_map": direct_index,
            "event_local_sync_index": _first_present(
                event_sync.get("event_local_sync_index"),
                ext_trigger.get("last_selected_sync_index"),
            ),
            "last_mapped_mvs_sync_index": mapped_i,
            "mapped_delta_to_bev_target": None if mapped_i is None or bev_target_i is None else int(mapped_i - bev_target_i),
            "all_edge_count": all_edges,
            "selected_edge_count": selected_edges,
            "dedupe_skipped_edges": to_int(ext_trigger.get("dedupe_skipped_edges"), 0),
            "sync_start_tick_id": _first_present(ext_trigger.get("sync_start_tick_id"), event_sync.get("sync_start_tick_id")),
            "map_count": event_sync.get("map_count"),
            "timeline_loaded_count": event_sync.get("timeline_loaded_count"),
            "last_error": _first_present(status.get("last_error"), event_sync.get("last_error")),
        }

    return {
        "diagnostic_scope": "event_worker_producer_head_latest",
        "display_sync_note": "Producer head can run ahead of BEV display when event_layer.presentation_worker selects historical bundles.",
        "bev_target_sync_id": bev_target_i,
        "camera_count": len(cams),
        "locked_cam_count": locked_count,
        "direct_index_cam_count": direct_index_count,
        "total_selected_edge_count": trigger_edge_count,
        "mapped_sync_min": min(mapped_values) if mapped_values else None,
        "mapped_sync_max": max(mapped_values) if mapped_values else None,
        "mapped_sync_span": (max(mapped_values) - min(mapped_values)) if mapped_values else None,
        "mapped_sync_delta_min_to_bev": (min(mapped_values) - bev_target_i) if mapped_values and bev_target_i is not None else None,
        "mapped_sync_delta_max_to_bev": (max(mapped_values) - bev_target_i) if mapped_values and bev_target_i is not None else None,
        "cameras": cams,
    }


def collect_bev_event_display_sync_diagnostics(bev: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize the Event layer frame that Global BEV actually displays.

    `event_worker_*_status.json` reports the producer head.  In V24/V25 the BEV
    renderer can deliberately consume an older event bundle from a history ring,
    so display-sync acceptance must be based on presentation_worker, not the
    latest producer head.
    """
    layer = bev.get("event_layer", {}) if isinstance(bev.get("event_layer"), dict) else {}
    presentation = layer.get("presentation_worker", {}) if isinstance(layer.get("presentation_worker"), dict) else {}
    alignment = bev.get("alignment", {}) if isinstance(bev.get("alignment"), dict) else {}
    quality = alignment.get("quality", {}) if isinstance(alignment.get("quality"), dict) else {}

    enabled = bool(presentation.get("enabled"))
    mode = str(presentation.get("mode") or "")
    state = str(presentation.get("state") or "")
    worker_state = str(presentation.get("worker_state") or "")
    bundle_lookup_state = str(presentation.get("bundle_lookup_state") or "")
    compare_lookup_state = str(presentation.get("compare_lookup_state") or "")
    display_compare_state = str(presentation.get("display_compare_state") or presentation.get("compare_state") or "")
    display_mismatch = to_int(
        _first_present(presentation.get("display_mismatch_cam_count"), presentation.get("mismatch_cam_count")),
        0,
    )
    bundle_hit_ratio = to_float(presentation.get("bundle_hit_ratio"), 0.0)
    compare_hit_ratio = to_float(presentation.get("compare_hit_ratio"), 0.0)
    bundle_delta_raw = _first_present(
        presentation.get("bundle_target_delta_ticks"),
        presentation.get("target_delta_ticks"),
        presentation.get("compare_target_delta_ticks"),
    )
    bundle_delta = None if bundle_delta_raw is None else to_int(bundle_delta_raw, 0)
    tolerance = to_int(presentation.get("compare_tolerance_ticks"), 1)
    tolerance = max(1, tolerance)

    lookup_ok = bundle_lookup_state in {"exact", "near"} or compare_lookup_state in {"exact", "near"}
    delta_ok = bundle_delta is not None and abs(int(bundle_delta)) <= tolerance
    display_compare_ok = display_compare_state in {"match", "pending_target", ""}
    hit_ratio_ok = bundle_hit_ratio >= 0.8 or compare_hit_ratio >= 0.8
    consumed = to_int(presentation.get("active_consume_count"), 0) > 0 or to_int(presentation.get("bundle_hit_count"), 0) > 0
    synced = bool(enabled and lookup_ok and delta_ok and display_mismatch == 0 and (hit_ratio_ok or consumed))
    if not enabled:
        state_s = "disabled"
    elif synced:
        state_s = "synced"
    elif lookup_ok and delta_ok:
        state_s = "near_but_unproven"
    elif bundle_lookup_state or compare_lookup_state:
        state_s = "not_synced"
    else:
        state_s = "missing"

    return {
        "diagnostic_scope": "global_bev_display_selected_event_bundle",
        "state": state_s,
        "synced": synced,
        "enabled": enabled,
        "mode": mode,
        "presentation_state": state,
        "worker_state": worker_state,
        "target_sync_id": _first_present(bev.get("target_sync_id"), alignment.get("target_sync_id")),
        "alignment_overall_state": alignment.get("overall_state"),
        "alignment_event_synced_cam_count": quality.get("event_synced_cam_count"),
        "bundle_lookup_state": bundle_lookup_state,
        "bundle_target_delta_ticks": bundle_delta,
        "compare_lookup_state": compare_lookup_state,
        "compare_target_delta_ticks": presentation.get("compare_target_delta_ticks"),
        "compare_tolerance_ticks": tolerance,
        "display_compare_state": display_compare_state,
        "display_mismatch_cam_count": display_mismatch,
        "bundle_hit_ratio": bundle_hit_ratio,
        "bundle_hit_count": presentation.get("bundle_hit_count"),
        "bundle_miss_count": presentation.get("bundle_miss_count"),
        "compare_hit_ratio": compare_hit_ratio,
        "consume_exact_count": presentation.get("consume_exact_count"),
        "consume_near_count": presentation.get("consume_near_count"),
        "active_consume_count": presentation.get("active_consume_count"),
        "fallback_count": presentation.get("fallback_count"),
        "cached_target_min": presentation.get("cached_target_min"),
        "cached_target_max": presentation.get("cached_target_max"),
        "updated_age_ms": presentation.get("updated_age_ms"),
    }


def compact_event_overlay_rows(overview: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = overview.get("event_overlay", {}) if isinstance(overview.get("event_overlay"), dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for cam, rec in rows.items():
        if not isinstance(rec, dict):
            continue
        out[str(cam)] = {
            "drawn": bool(rec.get("drawn", False)),
            "stale": bool(rec.get("stale", True)),
            "age_ms": rec.get("age_ms"),
            "reason": rec.get("reason"),
            "open_error": rec.get("open_error"),
            "reader_name": rec.get("reader_name"),
            "reader_path": rec.get("reader_path"),
            "reader_sidecar_path": rec.get("reader_sidecar_path"),
            "seq": rec.get("seq"),
            "shm_frame_id": rec.get("shm_frame_id"),
            "raw": rec.get("raw"),
            "filtered": rec.get("filtered"),
            "mask_valid": rec.get("mask_valid"),
        }
    return out


def _snapshot_overview_rows(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    overview = snapshot.get("event_overlay_overview", {}) if isinstance(snapshot.get("event_overlay_overview"), dict) else {}
    rows = overview.get("event_overlay", {}) if isinstance(overview.get("event_overlay"), dict) else {}
    return rows


def _best_overview_row(cam: str, *snapshots: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    best: Dict[str, Any] = {}
    best_src = ""
    best_score = -1.0
    for idx, snap in enumerate(snapshots):
        if not isinstance(snap, dict):
            continue
        rec = _snapshot_overview_rows(snap).get(cam)
        if not isinstance(rec, dict):
            continue
        age = to_float(rec.get("age_ms"), 1e9)
        drawn = 1.0 if bool(rec.get("drawn", False)) else 0.0
        stale = 1.0 if bool(rec.get("stale", True)) else 0.0
        score = drawn * 10000.0 - stale * 5000.0 - max(0.0, age) - idx
        if score > best_score:
            best = rec
            best_src = str(snap.get("tag", f"snapshot_{idx}"))
            best_score = score
    return best, best_src


def _worker_sidecar_evidence(
        run_dir: Path,
        cam: str,
        reference_wall_us: int = 0,
        min_wall_us: int = 0,
        max_wall_us: int = 0) -> Dict[str, Any]:
    """Find strong archived evidence that the Event worker was publishing.

    The final system-status sample can be collected while launcher shutdown is
    already in progress. In that case an Event worker status file may say
    "stopping" even though the runtime path was healthy. Worker-owned sidecars
    are preferred; EventPerf lines from the Event worker stdout are accepted as
    fallback evidence because they are emitted by the actual processing loop.
    """
    candidates = [
        run_dir / "status" / f"event_overlay_{cam}_latest.json",
        run_dir / "status" / f"event_viz_{cam}_latest.json",
    ]
    best: Dict[str, Any] = {}
    best_score = -1.0
    for path in candidates:
        rec = read_json(path)
        if rec.get("_error"):
            continue
        rec_cam = str(rec.get("camera", rec.get("cam", cam))).strip()
        if rec_cam and rec_cam != cam:
            continue
        published_us = to_int(rec.get("published_wall_us", rec.get("wall_us", rec.get("ts_wall_us"))), 0)
        if published_us <= 0:
            continue
        if min_wall_us > 0 and published_us < min_wall_us:
            continue
        if max_wall_us > 0 and published_us > max_wall_us:
            continue
        if reference_wall_us > 0 and abs(reference_wall_us - published_us) > 5_000_000:
            continue
        seq = to_int(rec.get("seq", rec.get("frame_seq", rec.get("shm_frame_id"))), 0)
        score = float(published_us) + float(seq)
        if score > best_score:
            best = {
                "camera": cam,
                "path": str(path),
                "published_wall_us": published_us,
                "seq": seq,
                "source_stage": rec.get("source_stage"),
                "raw_event_count": rec.get("raw_event_count"),
                "filtered_event_count": rec.get("filtered_event_count"),
            }
            best_score = score
    if best:
        return best

    log_path = run_dir / "launcher_logs" / "event.log"
    log_text = read_text(log_path, max_chars=2_000_000)
    marker = f"[EventPerf][{cam}]"
    sample_count = 0
    last_line = ""
    last_line_no = 0
    for idx, line in enumerate(log_text.splitlines(), 1):
        if marker not in line:
            continue
        sample_count += 1
        last_line = line.strip()
        last_line_no = idx
    if sample_count <= 0:
        return {}
    return {
        "camera": cam,
        "path": str(log_path),
        "source_stage": "event_worker_perf_log",
        "sample_count": int(sample_count),
        "line_no_in_tail": int(last_line_no),
        "last_line_tail": last_line[-360:],
    }


def reconcile_runtime_summary_with_overview(summary: Dict[str, Any], *snapshots: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Remove display-overlay false positives when a fresher overview row proves the camera is drawing.

    This intentionally only touches event overlay display links.  Worker, sync,
    MVS, YOLO, and hit-chain failures remain authoritative.
    """
    if not isinstance(summary, dict):
        return {}, []
    out = dict(summary)
    broken_in = list(out.get("broken_links", []) or [])
    degraded_in = list(out.get("degraded_links", []) or [])
    broken_out: List[Any] = []
    fixed: List[Dict[str, Any]] = []
    overlay_suffixes = {"overlay_stale", "overlay_not_drawn", "overlay_missing"}
    for link in broken_in:
        link_s = str(link)
        parts = link_s.split(":")
        if len(parts) == 3 and parts[0] == "event" and parts[2] in overlay_suffixes:
            cam = parts[1]
            rec, source = _best_overview_row(cam, *snapshots)
            age_ms = to_float(rec.get("age_ms"), 1e9) if rec else 1e9
            if rec and bool(rec.get("drawn", False)) and not bool(rec.get("stale", True)) and age_ms <= 1000.0:
                fixed.append({
                    "link": link_s,
                    "camera": cam,
                    "source": source,
                    "age_ms": age_ms,
                    "reader_name": rec.get("reader_name"),
                    "reader_path": rec.get("reader_path"),
                    "seq": rec.get("seq"),
                })
                continue
        broken_out.append(link)
    degraded_out = degraded_in
    if fixed and not any(str(x).startswith("event:") for x in broken_out):
        degraded_out = [x for x in degraded_out if str(x) != "event"]
    out["broken_links"] = broken_out
    out["degraded_links"] = degraded_out
    if broken_out:
        out["overall_state"] = "BROKEN"
    elif degraded_out:
        out["overall_state"] = "DEGRADED"
    elif str(out.get("overall_state", "")).upper() in {"BROKEN", "DEGRADED", "WARN", "WARNING", "OK"}:
        out["overall_state"] = "OK"
    if fixed:
        out["overview_reconciled_links"] = fixed
    return out, fixed


def reconcile_runtime_summary_with_worker_sidecars(
        summary: Dict[str, Any],
        run_dir: Path,
        *snapshots: Dict[str, Any],
        reference_wall_us: int = 0,
        min_wall_us: int = 0,
        max_wall_us: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Remove Event worker false positives only when archived worker evidence is strong."""
    if not isinstance(summary, dict):
        return {}, []
    out = dict(summary)
    broken_in = list(out.get("broken_links", []) or [])
    degraded_in = list(out.get("degraded_links", []) or [])
    broken_out: List[Any] = []
    fixed: List[Dict[str, Any]] = []
    for link in broken_in:
        link_s = str(link)
        parts = link_s.split(":")
        if len(parts) == 3 and parts[0] == "event" and parts[2] == "worker":
            cam = parts[1]
            rec, source = _best_overview_row(cam, *snapshots)
            age_ms = to_float(rec.get("age_ms"), 1e9) if rec else 1e9
            sidecar = _worker_sidecar_evidence(
                run_dir,
                cam,
                reference_wall_us=reference_wall_us,
                min_wall_us=min_wall_us,
                max_wall_us=max_wall_us,
            )
            if (
                    rec
                    and bool(rec.get("drawn", False))
                    and not bool(rec.get("stale", True))
                    and age_ms <= 1000.0
                    and sidecar):
                fixed.append({
                    "link": link_s,
                    "camera": cam,
                    "overview_source": source,
                    "overview_age_ms": age_ms,
                    "reader_name": rec.get("reader_name"),
                    "reader_path": rec.get("reader_path"),
                    "overview_seq": rec.get("seq"),
                    "worker_sidecar": sidecar,
                })
                continue
        broken_out.append(link)
    degraded_out = degraded_in
    if fixed and not any(str(x).startswith("event:") for x in broken_out):
        degraded_out = [x for x in degraded_out if str(x) != "event"]
    out["broken_links"] = broken_out
    out["degraded_links"] = degraded_out
    if broken_out:
        out["overall_state"] = "BROKEN"
    elif degraded_out:
        out["overall_state"] = "DEGRADED"
    elif str(out.get("overall_state", "")).upper() in {"BROKEN", "DEGRADED", "WARN", "WARNING", "OK"}:
        out["overall_state"] = "OK"
    if fixed:
        out["worker_reconciled_links"] = fixed
    return out, fixed


def collect_snapshot(project: Path, run_dir: Optional[Path], tag: str) -> Dict[str, Any]:
    sync = project / "sync_ipc"
    bev = read_json(sync / "global_bev_status.json")
    overview = read_json(sync / "event_overlay_overview_status.json")
    system = read_json(sync / "system_status_latest.json")
    yolo = read_json(sync / "global_yolo_status.json")
    trigger = read_json(sync / "trigger_status.json")
    hit = read_json(sync / "hit_judge_status.json")
    person_dataset = read_json(sync / "person_id_dataset_status.json")
    bev_control = read_json(sync / "global_bev_control.json")
    global_person_status = read_json(sync / "global_person_status.json")
    global_person_shadow_status = read_json(sync / "global_person_shadow_status.json")
    global_person_control = read_json(sync / "global_person_id_control.json")
    global_person_shadow_control = read_json(sync / "global_person_shadow_control.json")
    experiment_control = read_json(sync / "experiment_control.json")
    experiment_status = read_json(sync / "experiment_status_latest.json")
    event_layer, event_layer_source = first_dict(bev, "event_layer", "event_layer_overlay")
    event_bullet, event_bullet_source = first_dict(bev, "event_bullet", "event_bullet_overlay")
    global_person, global_person_source = first_dict(bev, "global_person", "global_person_overlay")
    alignment = bev.get("alignment", {}) if isinstance(bev.get("alignment"), dict) else {}

    payload: Dict[str, Any] = {
        "kind": "fullsys_snapshot",
        "tag": tag,
        "ts": time.time(),
        "project": str(project),
        "run_dir": str(run_dir) if run_dir else "",
        "global_bev": {
            "published_wall_us": bev.get("published_wall_us"),
            "published_wall_iso": bev.get("published_wall_iso"),
            "render_fps": bev.get("render_fps"),
            "output_fps": bev.get("output_fps"),
            "last_error": bev.get("last_error"),
            "age_ms": bev.get("_age_ms"),
            "target_sync_id": bev.get("target_sync_id"),
            "alignment": alignment,
            "mvs_texture_upload": bev.get("mvs_texture_upload"),
            "event_layer": event_layer,
            "event_layer_source": event_layer_source,
            "event_bullet": event_bullet,
            "event_bullet_source": event_bullet_source,
            "global_person": global_person,
            "global_person_source": global_person_source,
            "readback": bev.get("readback"),
        },
        "event_overlay_overview": {
            "age_ms": overview.get("_age_ms"),
            "control": overview.get("control"),
            "last_error": overview.get("last_error"),
            "overview_visible": overview.get("overview_visible"),
            "window": overview.get("window"),
            "dilate": overview.get("dilate"),
            "event_overlay": compact_event_overlay_rows(overview),
        },
        "event_worker_sync": collect_event_worker_sync_diagnostics(sync, bev),
        "event_display_sync": collect_bev_event_display_sync_diagnostics(bev),
        "system_status": {
            "age_ms": system.get("_age_ms"),
            "schema_version": system.get("schema_version"),
            "summary": system.get("summary"),
            "status_cn": system.get("status_cn"),
            "keys": sorted(system.keys())[:80] if isinstance(system, dict) else [],
        },
        "global_yolo": {
            "age_ms": yolo.get("_age_ms"),
            "worker_count": yolo.get("worker_count"),
            "fps": yolo.get("fps"),
            "inferred_fps_total": yolo.get("inferred_fps_total"),
            "published_fps_total": yolo.get("published_fps_total"),
            "async_record_postprocess_enabled": yolo.get("async_record_postprocess_enabled"),
            "postprocess_queue_depth": yolo.get("postprocess_queue_depth"),
            "postprocess_lag_ms": yolo.get("postprocess_lag_ms"),
            "postprocess_backpressure_ms": yolo.get("postprocess_backpressure_ms"),
            "postprocess_dropped_batches": yolo.get("postprocess_dropped_batches"),
            "polygon_contract_ok": yolo.get("polygon_contract_ok"),
            "published_records_with_polygon_ratio": yolo.get("published_records_with_polygon_ratio"),
            "published_persons_with_polygon_ratio": yolo.get("published_persons_with_polygon_ratio"),
            "input_alignment_audit": yolo.get("input_alignment_audit"),
            "input_buffer": yolo.get("input_buffer"),
            "summary": yolo.get("summary"),
        },
        "global_person_id_status": {
            "age_ms": global_person_status.get("_age_ms"),
            "stream_role": global_person_status.get("stream_role"),
            "filter_backend": global_person_status.get("filter_backend"),
            "track_count": global_person_status.get("track_count"),
            "runtime_config": global_person_status.get("runtime_config"),
            "control": global_person_status.get("control"),
            "stats": global_person_status.get("stats"),
        },
        "global_person_id_shadow_status": {
            "age_ms": global_person_shadow_status.get("_age_ms"),
            "stream_role": global_person_shadow_status.get("stream_role"),
            "filter_backend": global_person_shadow_status.get("filter_backend"),
            "track_count": global_person_shadow_status.get("track_count"),
            "runtime_config": global_person_shadow_status.get("runtime_config"),
            "control": global_person_shadow_status.get("control"),
            "stats": global_person_shadow_status.get("stats"),
        },
        "global_person_id_control": global_person_control,
        "global_person_id_shadow_control": global_person_shadow_control,
        "trigger": {
            "age_ms": trigger.get("_age_ms"),
            "fps": trigger.get("fps"),
            "trigger_fps": trigger.get("trigger_fps"),
        },
        "hit_judge": {
            "age_ms": hit.get("_age_ms"),
            "enabled": hit.get("enabled"),
            "status": hit.get("status"),
        },
        "person_id_dataset": {
            "age_ms": person_dataset.get("_age_ms"),
            "state": person_dataset.get("state"),
            "recording": person_dataset.get("recording"),
            "dataset_kind": person_dataset.get("dataset_kind"),
            "dataset_type": person_dataset.get("dataset_type"),
            "session_id": person_dataset.get("session_id"),
            "session_dir": person_dataset.get("session_dir"),
            "counts": person_dataset.get("counts"),
            "last_error": person_dataset.get("last_error"),
        },
        "experiment": {
            "age_ms": experiment_status.get("_age_ms"),
            "experiment_id": experiment_control.get("experiment_id", experiment_status.get("experiment_id", "")),
            "stage": experiment_control.get("stage", experiment_status.get("stage", "")),
            "config_hash": experiment_control.get("config_hash", experiment_status.get("config_hash", "")),
            "state": experiment_status.get("state"),
            "control": experiment_control,
            "status": experiment_status,
        },
        "mask_stats_jsonl": mask_stats(project),
        "status_key_audit": collect_status_key_audit(bev, overview, system, hit, bev_control),
    }

    if run_dir:
        out = run_dir / "control_checks" / f"{tag}.json"
        write_json(out, payload)
    return payload


def _resolve_run_dir(project: Path, label: str = "", run_dir: str = "") -> Optional[Path]:
    if str(run_dir or "").strip():
        raw = Path(str(run_dir).strip()).expanduser()
        return raw.resolve() if raw.is_absolute() else (project / raw).resolve()
    sync = project / "sync_ipc"
    candidates: List[Path] = []
    if str(label or "").strip():
        exact = sync / f"latest_{label}_run_dir.txt"
        if exact.exists():
            candidates.append(exact)
    candidates.extend(sorted(sync.glob("latest_*_run_dir.txt"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True))
    seen = set()
    for latest in candidates:
        if latest in seen:
            continue
        seen.add(latest)
        try:
            text = latest.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        raw = Path(text).expanduser()
        path = raw.resolve() if raw.is_absolute() else (project / raw).resolve()
        if path.exists():
            return path
    return None


def _run_state(run_dir: Optional[Path]) -> Dict[str, Any]:
    if run_dir is None:
        return {"state": "UNKNOWN", "running": False, "run_dir": "", "launcher_rc": ""}
    running = (run_dir / "launcher_running.flag").exists()
    rc_path = run_dir / "launcher_rc.txt"
    rc = read_text(rc_path, max_chars=200).strip() if rc_path.exists() else ""
    if running:
        state = "RUNNING"
    elif rc:
        state = "FINISHED"
    else:
        state = "STARTING_OR_STALE"
    return {
        "state": state,
        "running": bool(running),
        "run_dir": str(run_dir),
        "launcher_rc": rc,
        "archive_exists": Path(str(run_dir) + ".tar.gz").exists(),
        "stop_requested": (run_dir / "stop_requested.json").exists(),
    }


def _compact_live_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    system = snapshot.get("system_status", {}) if isinstance(snapshot.get("system_status"), dict) else {}
    summary = system.get("summary", {}) if isinstance(system.get("summary"), dict) else {}
    bev = snapshot.get("global_bev", {}) if isinstance(snapshot.get("global_bev"), dict) else {}
    yolo = snapshot.get("global_yolo", {}) if isinstance(snapshot.get("global_yolo"), dict) else {}
    person = snapshot.get("global_person_id_status", {}) if isinstance(snapshot.get("global_person_id_status"), dict) else {}
    person_shadow = snapshot.get("global_person_id_shadow_status", {}) if isinstance(snapshot.get("global_person_id_shadow_status"), dict) else {}
    hit = snapshot.get("hit_judge", {}) if isinstance(snapshot.get("hit_judge"), dict) else {}
    dataset = snapshot.get("person_id_dataset", {}) if isinstance(snapshot.get("person_id_dataset"), dict) else {}
    layer = bev.get("event_layer", {}) if isinstance(bev.get("event_layer"), dict) else {}
    layer_presentation = layer.get("presentation_worker", {}) if isinstance(layer.get("presentation_worker"), dict) else {}
    alignment = bev.get("alignment", {}) if isinstance(bev.get("alignment"), dict) else {}
    align_quality = alignment.get("quality", {}) if isinstance(alignment.get("quality"), dict) else {}
    audit = snapshot.get("status_key_audit", {}) if isinstance(snapshot.get("status_key_audit"), dict) else {}
    event_worker_sync = snapshot.get("event_worker_sync", {}) if isinstance(snapshot.get("event_worker_sync"), dict) else {}
    event_display_sync = snapshot.get("event_display_sync", {}) if isinstance(snapshot.get("event_display_sync"), dict) else {}
    return {
        "overall_state": summary.get("overall_state"),
        "broken_links": summary.get("broken_links", []),
        "degraded_links": summary.get("degraded_links", []),
        "status_key_ok": audit.get("ok"),
        "event_worker_sync": {
            "diagnostic_scope": event_worker_sync.get("diagnostic_scope"),
            "display_sync_note": event_worker_sync.get("display_sync_note"),
            "bev_target_sync_id": event_worker_sync.get("bev_target_sync_id"),
            "locked_cam_count": event_worker_sync.get("locked_cam_count"),
            "direct_index_cam_count": event_worker_sync.get("direct_index_cam_count"),
            "total_selected_edge_count": event_worker_sync.get("total_selected_edge_count"),
            "mapped_sync_min": event_worker_sync.get("mapped_sync_min"),
            "mapped_sync_max": event_worker_sync.get("mapped_sync_max"),
            "mapped_sync_span": event_worker_sync.get("mapped_sync_span"),
            "mapped_sync_delta_min_to_bev": event_worker_sync.get("mapped_sync_delta_min_to_bev"),
            "mapped_sync_delta_max_to_bev": event_worker_sync.get("mapped_sync_delta_max_to_bev"),
            "cameras": event_worker_sync.get("cameras", {}),
        },
        "event_display_sync": {
            "diagnostic_scope": event_display_sync.get("diagnostic_scope"),
            "state": event_display_sync.get("state"),
            "synced": event_display_sync.get("synced"),
            "enabled": event_display_sync.get("enabled"),
            "mode": event_display_sync.get("mode"),
            "presentation_state": event_display_sync.get("presentation_state"),
            "worker_state": event_display_sync.get("worker_state"),
            "target_sync_id": event_display_sync.get("target_sync_id"),
            "alignment_overall_state": event_display_sync.get("alignment_overall_state"),
            "alignment_event_synced_cam_count": event_display_sync.get("alignment_event_synced_cam_count"),
            "bundle_lookup_state": event_display_sync.get("bundle_lookup_state"),
            "bundle_target_delta_ticks": event_display_sync.get("bundle_target_delta_ticks"),
            "compare_lookup_state": event_display_sync.get("compare_lookup_state"),
            "compare_target_delta_ticks": event_display_sync.get("compare_target_delta_ticks"),
            "compare_tolerance_ticks": event_display_sync.get("compare_tolerance_ticks"),
            "display_compare_state": event_display_sync.get("display_compare_state"),
            "display_mismatch_cam_count": event_display_sync.get("display_mismatch_cam_count"),
            "bundle_hit_ratio": event_display_sync.get("bundle_hit_ratio"),
            "bundle_hit_count": event_display_sync.get("bundle_hit_count"),
            "bundle_miss_count": event_display_sync.get("bundle_miss_count"),
            "compare_hit_ratio": event_display_sync.get("compare_hit_ratio"),
            "consume_exact_count": event_display_sync.get("consume_exact_count"),
            "consume_near_count": event_display_sync.get("consume_near_count"),
            "active_consume_count": event_display_sync.get("active_consume_count"),
            "fallback_count": event_display_sync.get("fallback_count"),
            "cached_target_min": event_display_sync.get("cached_target_min"),
            "cached_target_max": event_display_sync.get("cached_target_max"),
            "updated_age_ms": event_display_sync.get("updated_age_ms"),
        },
        "global_bev": {
            "published_wall_us": bev.get("published_wall_us"),
            "render_fps": bev.get("render_fps"),
            "output_fps": bev.get("output_fps"),
            "target_sync_id": bev.get("target_sync_id"),
            "event_layer_dilate_px": layer.get("dilate_px"),
            "event_layer_dilate_backend": layer.get("dilate_backend"),
            "mvs_texture_upload": bev.get("mvs_texture_upload"),
            "event_layer_display_budget_ms": layer.get("display_budget_ms"),
            "event_layer_display_budget_elapsed_ms": layer.get("display_budget_elapsed_ms"),
            "event_layer_display_budget_held_cam_count": layer.get("display_budget_held_cam_count"),
            "event_layer_display_budget_overrun_rate": layer.get("display_budget_overrun_rate"),
            "event_layer_display_budget_order": layer.get("display_budget_order"),
            "event_layer_presentation_worker": {
                "enabled": layer_presentation.get("enabled"),
                "mode": layer_presentation.get("mode"),
                "state": layer_presentation.get("state"),
                "worker_state": layer_presentation.get("worker_state"),
                "payload_policy": layer_presentation.get("payload_policy"),
                "read_points": layer_presentation.get("read_points"),
                "compare_state": layer_presentation.get("compare_state"),
                "display_compare_state": layer_presentation.get("display_compare_state"),
                "delta_compare_state": layer_presentation.get("delta_compare_state"),
                "mismatch_cam_count": layer_presentation.get("mismatch_cam_count"),
                "display_mismatch_cam_count": layer_presentation.get("display_mismatch_cam_count"),
                "delta_mismatch_cam_count": layer_presentation.get("delta_mismatch_cam_count"),
                "compare_lookup_state": layer_presentation.get("compare_lookup_state"),
                "compare_target_delta_ticks": layer_presentation.get("compare_target_delta_ticks"),
                "compare_tolerance_ticks": layer_presentation.get("compare_tolerance_ticks"),
                "compare_exact_count": layer_presentation.get("compare_exact_count"),
                "compare_near_count": layer_presentation.get("compare_near_count"),
                "compare_miss_count": layer_presentation.get("compare_miss_count"),
                "compare_hit_ratio": layer_presentation.get("compare_hit_ratio"),
                "target_delta_ticks": layer_presentation.get("target_delta_ticks"),
                "updated_age_ms": layer_presentation.get("updated_age_ms"),
                "build_elapsed_ms": layer_presentation.get("build_elapsed_ms"),
                "build_p95_ms": layer_presentation.get("build_p95_ms"),
                "active_consume_elapsed_ms": layer_presentation.get("active_consume_elapsed_ms"),
                "active_consume_count": layer_presentation.get("active_consume_count"),
                "fallback_count": layer_presentation.get("fallback_count"),
                "fallback_reason": layer_presentation.get("fallback_reason"),
                "last_fallback_reason": layer_presentation.get("last_fallback_reason"),
                "bundle_lookup_state": layer_presentation.get("bundle_lookup_state"),
                "bundle_target_delta_ticks": layer_presentation.get("bundle_target_delta_ticks"),
                "bundle_hit_count": layer_presentation.get("bundle_hit_count"),
                "bundle_miss_count": layer_presentation.get("bundle_miss_count"),
                "bundle_hit_ratio": layer_presentation.get("bundle_hit_ratio"),
                "consume_exact_count": layer_presentation.get("consume_exact_count"),
                "consume_near_count": layer_presentation.get("consume_near_count"),
                "bundle_ring_size": layer_presentation.get("bundle_ring_size"),
                "bundle_ring_count": layer_presentation.get("bundle_ring_count"),
                "cached_target_min": layer_presentation.get("cached_target_min"),
                "cached_target_max": layer_presentation.get("cached_target_max"),
            },
            "alignment": {
                "mode": alignment.get("mode"),
                "state": alignment.get("state"),
                "overall_state": alignment.get("overall_state"),
                "target_sync_id": alignment.get("target_sync_id"),
                "delay_ms": alignment.get("delay_ms"),
                "mvs_synced_cam_count": align_quality.get("mvs_synced_cam_count"),
                "mvs_missing_history_cam_count": align_quality.get("mvs_missing_history_cam_count"),
                "mvs_metadata_outlier_cam_count": align_quality.get("mvs_metadata_outlier_cam_count"),
                "event_synced_cam_count": align_quality.get("event_synced_cam_count"),
                "event_sync_outlier_cam_count": align_quality.get("event_sync_outlier_cam_count"),
                "worker_elapsed_p95_ms": alignment.get("worker_elapsed_p95_ms"),
                "gui_refresh_p95_ms": alignment.get("gui_refresh_p95_ms"),
                "gui_upload_selected_p95_ms": alignment.get("gui_upload_selected_p95_ms"),
                "gui_upload_event_layer_p95_ms": alignment.get("gui_upload_event_layer_p95_ms"),
                "gui_paint_total_p95_ms": alignment.get("gui_paint_total_p95_ms"),
                "mvs_upload_budget_ms": alignment.get("mvs_upload_budget_ms"),
                "mvs_upload_elapsed_ms": alignment.get("mvs_upload_elapsed_ms"),
                "mvs_upload_held_by_budget_cam_count": alignment.get("mvs_upload_held_by_budget_cam_count"),
                "event_layer_display_budget_ms": alignment.get("event_layer_display_budget_ms"),
                "event_layer_display_budget_elapsed_ms": alignment.get("event_layer_display_budget_elapsed_ms"),
                "event_layer_display_budget_held_cam_count": alignment.get("event_layer_display_budget_held_cam_count"),
                "event_layer_display_budget_overrun_rate": alignment.get("event_layer_display_budget_overrun_rate"),
                "mvs_upload_budget_overrun_rate": alignment.get("mvs_upload_budget_overrun_rate"),
                "gui_event_sparse_read_p95_ms": alignment.get("gui_event_sparse_read_p95_ms"),
                "gui_event_sparse_sync_gate_p95_ms": alignment.get("gui_event_sparse_sync_gate_p95_ms"),
                "gui_event_sparse_project_p95_ms": alignment.get("gui_event_sparse_project_p95_ms"),
                "mvs_offset_ms_by_cam": alignment.get("mvs_offset_ms_by_cam"),
                "event_delta_ms_by_cam": alignment.get("event_delta_ms_by_cam"),
            },
            "last_error": bev.get("last_error"),
        },
        "global_yolo": {
            "worker_count": yolo.get("worker_count"),
            "fps": yolo.get("fps"),
            "inferred_fps_total": yolo.get("inferred_fps_total"),
            "published_fps_total": yolo.get("published_fps_total"),
            "async_record_postprocess_enabled": yolo.get("async_record_postprocess_enabled"),
            "postprocess_queue_depth": yolo.get("postprocess_queue_depth"),
            "postprocess_lag_ms": yolo.get("postprocess_lag_ms"),
            "postprocess_backpressure_ms": yolo.get("postprocess_backpressure_ms"),
            "postprocess_dropped_batches": yolo.get("postprocess_dropped_batches"),
            "polygon_contract_ok": yolo.get("polygon_contract_ok"),
            "published_records_with_polygon_ratio": yolo.get("published_records_with_polygon_ratio"),
            "published_persons_with_polygon_ratio": yolo.get("published_persons_with_polygon_ratio"),
            "input_alignment_audit": yolo.get("input_alignment_audit"),
            "input_buffer": yolo.get("input_buffer"),
            "summary": yolo.get("summary"),
        },
        "global_person_id": {
            "track_count": person.get("track_count"),
            "filter_backend": person.get("filter_backend"),
            "created_tracks": (person.get("stats", {}) if isinstance(person.get("stats"), dict) else {}).get("created_tracks"),
            "id_switch_suspects": (person.get("stats", {}) if isinstance(person.get("stats"), dict) else {}).get("id_switch_suspects"),
            "shadow_track_count": person_shadow.get("track_count"),
            "shadow_filter_backend": person_shadow.get("filter_backend"),
        },
        "hit_judge": {
            "enabled": hit.get("enabled"),
            "status": hit.get("status"),
        },
        "person_id_dataset": {
            "state": dataset.get("state"),
            "recording": dataset.get("recording"),
            "dataset_kind": dataset.get("dataset_kind"),
            "dataset_type": dataset.get("dataset_type"),
            "session_id": dataset.get("session_id"),
            "human_person_rows_total": (dataset.get("counts", {}) if isinstance(dataset.get("counts"), dict) else {}).get("human_person_rows_total"),
            "bullet_point_rows": (dataset.get("counts", {}) if isinstance(dataset.get("counts"), dict) else {}).get("bullet_point_rows"),
            "pseudo_hit_rows": (dataset.get("counts", {}) if isinstance(dataset.get("counts"), dict) else {}).get("pseudo_hit_rows"),
        },
    }


def collect_live_status(project: Path, run_dir: Optional[Path], label: str, tag: str = "live") -> Dict[str, Any]:
    sync = project / "sync_ipc"
    snapshot = collect_snapshot(project, None, tag)
    experiment_control = read_json(sync / "experiment_control.json")
    experiment_status = read_json(sync / "experiment_status_latest.json")
    payload = {
        "schema_version": 1,
        "kind": "test_run_status_latest",
        "tag": tag,
        "label": str(label or ""),
        "ts_wall_us": now_wall_us(),
        "project": str(project),
        "run": _run_state(run_dir),
        "compact": _compact_live_snapshot(snapshot),
        "experiment": {
            "control": experiment_control,
            "status": experiment_status,
            "experiment_id": experiment_control.get("experiment_id", experiment_status.get("experiment_id", "")),
            "stage": experiment_control.get("stage", experiment_status.get("stage", "")),
            "config_hash": experiment_control.get("config_hash", experiment_status.get("config_hash", "")),
        },
        "status_contract": {
            "machine_readable": True,
            "primary_files": [
                "sync_ipc/test_run_status_latest.json",
                "sync_ipc/experiment_status_latest.json",
                "sync_ipc/system_status_latest.json",
            ],
            "terminal_output": "non_authoritative_ascii_trigger_only",
        },
    }
    return payload


def compact_live_event(status: Dict[str, Any]) -> Dict[str, Any]:
    experiment = status.get("experiment", {}) if isinstance(status.get("experiment"), dict) else {}
    return {
        "schema_version": 1,
        "kind": "test_run_status_event",
        "tag": status.get("tag"),
        "label": status.get("label"),
        "ts_wall_us": status.get("ts_wall_us"),
        "run": status.get("run"),
        "experiment": {
            "experiment_id": experiment.get("experiment_id"),
            "stage": experiment.get("stage"),
            "config_hash": experiment.get("config_hash"),
        },
        "compact": status.get("compact", {}),
    }


def write_live_status(project: Path, run_dir: Optional[Path], status: Dict[str, Any]) -> None:
    sync = project / "sync_ipc"
    event = compact_live_event(status)
    write_json(sync / "test_run_status_latest.json", status)
    append_jsonl(sync / "test_run_events.jsonl", event)
    if run_dir is not None:
        write_json(run_dir / "status" / "test_run_status_latest.json", status)
        append_jsonl(run_dir / "test_run_status.jsonl", event)


def _load_profile(path_text: str) -> Tuple[Dict[str, Any], str]:
    text = str(path_text or "").strip()
    if not text:
        return {}, ""
    path = Path(text).expanduser()
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return (obj if isinstance(obj, dict) else {}), str(path)
    except Exception as exc:
        raise SystemExit(f"failed to load profile {text}: {type(exc).__name__}: {exc}")


def _subset(payload: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
    return {key: payload[key] for key in allowed if key in payload}


def set_experiment(project: Path, source: str, experiment_id: str = "", stage: str = "",
                   note: str = "", hit_profile_path: str = "", global_id_profile_path: str = "",
                   authority_mode: str = "", human_noise_mode: str = "",
                   bbox_start_expand_px: str = "", bbox_start_policy: str = "",
                   global_person_target: str = "",
                   global_person_filter_backend: str = "",
                   projected_mask_final_mode: str = "",
                   projected_mask_require_terminal_or_turn: str = "",
                   same_track_episode_dedup_ms: str = "",
                   project_extend_px: str = "",
                   project_max_last_dist_px: str = "") -> Dict[str, Any]:
    sync = project / "sync_ipc"
    hit_profile, hit_profile_loaded = _load_profile(hit_profile_path)
    global_profile, global_profile_loaded = _load_profile(global_id_profile_path)
    exp_id = str(experiment_id or hit_profile.get("experiment_id") or global_profile.get("experiment_id") or time.strftime("exp_%Y%m%d_%H%M%S"))
    stage_s = str(stage or hit_profile.get("stage") or global_profile.get("stage") or "")

    hit_allowed = [
        "final_hit_authority",
        "strong_gate_enabled",
        "strong_reasoning_enabled",
        "strong_reasoning_shadow_only",
        "pseudo_hit_enable",
        "human_noise_filter_enable",
        "human_noise_filter_mode",
        "human_noise_soft_threshold",
        "human_noise_hard_threshold",
        "human_noise_track_start_bbox_filter_enable",
        "human_noise_track_start_bbox_expand_px",
        "human_noise_track_start_bbox_policy",
        "final_hit_cross_camera_dedup_ms",
        "final_hit_same_track_episode_dedup_enable",
        "final_hit_same_track_episode_dedup_ms",
        "projected_mask_final_mode",
        "projected_mask_require_terminal_or_turn",
        "strong_reasoning_track_project_enable",
        "strong_reasoning_track_project_extend_px",
        "strong_reasoning_track_project_max_last_dist_px",
    ]
    hit_payload = _subset(hit_profile, hit_allowed)
    if authority_mode:
        if authority_mode not in {"strong_gate", "strong_reasoning", "both_shadow_compare"}:
            raise SystemExit(f"unsupported authority mode: {authority_mode}")
        hit_payload["final_hit_authority"] = authority_mode
        if authority_mode == "strong_reasoning":
            hit_payload.setdefault("strong_reasoning_shadow_only", False)
    if human_noise_mode:
        if human_noise_mode not in {"off", "shadow", "soft", "hard"}:
            raise SystemExit(f"unsupported human noise mode: {human_noise_mode}")
        hit_payload["human_noise_filter_mode"] = human_noise_mode
        hit_payload["human_noise_filter_enable"] = human_noise_mode != "off"
    if str(bbox_start_expand_px or "").strip():
        hit_payload["human_noise_track_start_bbox_filter_enable"] = True
        hit_payload["human_noise_track_start_bbox_expand_px"] = float(bbox_start_expand_px)
    policy = str(bbox_start_policy or "").strip().lower()
    if policy:
        if policy not in {"soft", "hard", "off"}:
            raise SystemExit(f"unsupported bbox start policy: {policy}")
        hit_payload["human_noise_track_start_bbox_policy"] = policy
    mode = str(projected_mask_final_mode or "").strip().lower()
    if mode:
        if mode not in {"pending_only", "shadow_only", "final_allowed"}:
            raise SystemExit(f"unsupported projected mask final mode: {mode}")
        hit_payload["projected_mask_final_mode"] = mode
    if str(projected_mask_require_terminal_or_turn or "").strip():
        hit_payload["projected_mask_require_terminal_or_turn"] = parse_bool(projected_mask_require_terminal_or_turn)
    if str(same_track_episode_dedup_ms or "").strip():
        hit_payload["final_hit_same_track_episode_dedup_enable"] = True
        hit_payload["final_hit_same_track_episode_dedup_ms"] = float(same_track_episode_dedup_ms)
    if str(project_extend_px or "").strip():
        hit_payload["strong_reasoning_track_project_extend_px"] = float(project_extend_px)
    if str(project_max_last_dist_px or "").strip():
        hit_payload["strong_reasoning_track_project_max_last_dist_px"] = float(project_max_last_dist_px)
    if hit_payload:
        hit_payload.setdefault("final_hit_cross_camera_dedup_ms", 700.0)
        hit_payload.setdefault("final_hit_same_track_episode_dedup_enable", True)
        hit_payload.setdefault("final_hit_same_track_episode_dedup_ms", 1500.0)
        hit_payload.setdefault("projected_mask_final_mode", "pending_only")
        hit_payload.setdefault("projected_mask_require_terminal_or_turn", True)
        hit_payload.setdefault("strong_reasoning_track_project_extend_px", 100.0)
        hit_payload.setdefault("strong_reasoning_track_project_max_last_dist_px", 40.0)

    global_allowed = [
        "filter_backend",
        "count_gate_enable",
        "cluster_gate_m",
        "match_gate_m",
        "lost_match_gate_m",
        "reid_memory_ms",
        "retire_after_ms",
        "duplicate_suppress_enable",
        "duplicate_suppress_dist_m",
        "birth_quarantine_enable",
        "birth_confirm_hits",
        "birth_match_gate_m",
        "birth_near_track_suppress_m",
        "birth_candidate_ttl_ms",
        "handoff_aware_enable",
        "handoff_match_gate_m",
        "handoff_birth_near_track_suppress_m",
        "handoff_strict_count_lock_enable",
        "handoff_strict_birth_block_m",
        "kalman_use_for_association",
        "kalman_process_noise_m",
        "kalman_measurement_noise_m",
    ]
    gpid_payload = _subset(global_profile, global_allowed)
    if global_person_filter_backend:
        if global_person_filter_backend not in {"alpha_beta", "kalman_cv"}:
            raise SystemExit(f"unsupported global person filter backend: {global_person_filter_backend}")
        gpid_payload["filter_backend"] = global_person_filter_backend
    target = str(global_person_target or global_profile.get("target") or "shadow").strip().lower()
    if target not in {"official", "shadow", "both"}:
        raise SystemExit(f"unsupported global person target: {target}")

    applied: Dict[str, Any] = {}
    if hit_payload:
        applied["hit_judge_control"] = _merge_control_json(sync / "hit_judge_control.json", hit_payload, source)
    if gpid_payload:
        targets = ["official", "shadow"] if target == "both" else [target]
        for item in targets:
            path = sync / ("global_person_id_control.json" if item == "official" else "global_person_shadow_control.json")
            applied[f"global_person_id_{item}_control"] = _merge_control_json(path, gpid_payload, source)

    control = {
        "schema_version": 1,
        "kind": "experiment_control",
        "experiment_id": exp_id,
        "stage": stage_s,
        "note": note,
        "source": source,
        "updated_wall_us": now_wall_us(),
        "hit_profile_path": hit_profile_loaded,
        "global_id_profile_path": global_profile_loaded,
        "hit_profile": hit_profile,
        "global_id_profile": global_profile,
        "hit_payload": hit_payload,
        "global_person_payload": gpid_payload,
        "global_person_target": target,
    }
    control["config_hash"] = stable_hash(control)
    write_json(sync / "experiment_control.json", control)
    status = {
        "schema_version": 1,
        "kind": "experiment_status_latest",
        "state": "APPLIED",
        "experiment_id": exp_id,
        "stage": stage_s,
        "config_hash": control["config_hash"],
        "source": source,
        "updated_wall_us": now_wall_us(),
        "applied": applied,
        "control": control,
    }
    write_json(sync / "experiment_status_latest.json", status)
    append_jsonl(sync / "experiment_events.jsonl", status)
    return status


def _read_control_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _merge_control_json(path: Path, payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    current = _read_control_json(path)
    out = dict(current)
    out.update(payload)
    out["schema_version"] = 1
    out["seq"] = int(current.get("seq", 0) or 0) + 1
    out["source"] = source
    out["updated_wall_us"] = now_wall_us()
    write_json(path, out)
    return out


def set_control(project: Path, overview_visible: Optional[bool], bev_dilate_px: Optional[float],
                show_event_layer: Optional[bool], hit_enabled: Optional[bool],
                raw_record_cmd: str, raw_record_targets: str, raw_record_session_id: str,
                person_dataset_cmd: str, person_dataset_type: str,
                person_dataset_label: str, person_dataset_session_id: str, source: str,
                person_dataset_kind: str = "",
                person_dataset_include_mask_polygons: str = "",
                person_dataset_include_bullet_trajectory: str = "",
                person_dataset_include_hit_debug: str = "",
                final_hit_authority: str = "",
                strong_gate_enabled: str = "",
                strong_reasoning_enabled: str = "",
                strong_reasoning_shadow_only: str = "",
                human_noise_filter_enable: str = "",
                human_noise_filter_mode: str = "",
                human_noise_soft_threshold: str = "",
                human_noise_hard_threshold: str = "",
                human_noise_track_start_bbox_filter_enable: str = "",
                human_noise_track_start_bbox_expand_px: str = "",
                human_noise_track_start_bbox_policy: str = "",
                final_hit_cross_camera_dedup_ms: str = "",
                projected_mask_final_mode: str = "",
                projected_mask_require_terminal_or_turn: str = "",
                final_hit_same_track_episode_dedup_enable: str = "",
                final_hit_same_track_episode_dedup_ms: str = "",
                strong_reasoning_track_project_enable: str = "",
                strong_reasoning_track_project_extend_px: str = "",
                strong_reasoning_track_project_max_last_dist_px: str = "",
                global_person_target: str = "",
                global_person_filter_backend: str = "",
                global_person_reset_tracks: bool = False,
                global_person_cluster_gate_m: str = "",
                global_person_match_gate_m: str = "",
                global_person_lost_match_gate_m: str = "",
                global_person_reid_memory_ms: str = "",
                global_person_retire_after_ms: str = "",
                global_person_duplicate_suppress_dist_m: str = "",
                global_person_birth_confirm_hits: str = "",
                global_person_birth_match_gate_m: str = "",
                global_person_birth_near_track_suppress_m: str = "",
                global_person_birth_candidate_ttl_ms: str = "",
                global_person_birth_quarantine_enable: str = "",
                global_person_kalman_use_for_association: str = "",
                global_person_kalman_process_noise_m: str = "",
                global_person_kalman_measurement_noise_m: str = "") -> Dict[str, Any]:
    sync = project / "sync_ipc"
    changed: Dict[str, Any] = {"kind": "fullsys_control_update", "ts": time.time(), "source": source}

    if overview_visible is not None:
        overview_path = sync / "event_overlay_overview_control.json"
        overview = _merge_control_json(overview_path, {
            "event_overlay_overview_visible": bool(overview_visible),
        }, source)
        changed["event_overlay_overview_control"] = overview

    if bev_dilate_px is not None or show_event_layer is not None:
        bev_path = sync / "global_bev_control.json"
        bev: Dict[str, Any] = {}
        if show_event_layer is not None:
            bev["show_event_layer"] = bool(show_event_layer)
            bev["event_layer_enable"] = bool(show_event_layer)
        if bev_dilate_px is not None:
            bev["event_layer_dilate_px"] = float(bev_dilate_px)
        bev = _merge_control_json(bev_path, bev, source)
        changed["global_bev_control"] = bev

    if hit_enabled is not None:
        hit_path = sync / "hit_judge_control.json"
        hit = _merge_control_json(hit_path, {
            "hit_detection_enabled": bool(hit_enabled),
            "hit_candidate_output_enabled": bool(hit_enabled),
        }, source)
        changed["hit_judge_control"] = hit
    hit_chain_payload: Dict[str, Any] = {}
    authority = str(final_hit_authority or "").strip().lower()
    if authority:
        if authority not in {"strong_gate", "strong_reasoning", "both_shadow_compare"}:
            raise SystemExit(f"unsupported final hit authority: {authority}")
        hit_chain_payload["final_hit_authority"] = authority
        if authority == "strong_reasoning" and str(strong_reasoning_shadow_only or "").strip() == "":
            hit_chain_payload["strong_reasoning_shadow_only"] = False
    if str(strong_gate_enabled or "").strip() != "":
        hit_chain_payload["strong_gate_enabled"] = parse_bool(strong_gate_enabled)
    if str(strong_reasoning_enabled or "").strip() != "":
        hit_chain_payload["strong_reasoning_enabled"] = parse_bool(strong_reasoning_enabled)
        hit_chain_payload["pseudo_hit_enable"] = parse_bool(strong_reasoning_enabled)
    if str(strong_reasoning_shadow_only or "").strip() != "":
        hit_chain_payload["strong_reasoning_shadow_only"] = parse_bool(strong_reasoning_shadow_only)
    if hit_chain_payload:
        hit_path = sync / "hit_judge_control.json"
        hit = _merge_control_json(hit_path, hit_chain_payload, source)
        changed["hit_judge_control"] = hit
    noise_payload: Dict[str, Any] = {}
    if str(human_noise_filter_enable or "").strip() != "":
        noise_payload["human_noise_filter_enable"] = parse_bool(human_noise_filter_enable)
    mode = str(human_noise_filter_mode or "").strip().lower()
    if mode:
        if mode not in {"off", "shadow", "soft", "hard"}:
            raise SystemExit(f"unsupported human noise filter mode: {mode}")
        noise_payload["human_noise_filter_mode"] = mode
    for key, raw in (
        ("human_noise_soft_threshold", human_noise_soft_threshold),
        ("human_noise_hard_threshold", human_noise_hard_threshold),
        ("human_noise_track_start_bbox_expand_px", human_noise_track_start_bbox_expand_px),
        ("final_hit_cross_camera_dedup_ms", final_hit_cross_camera_dedup_ms),
        ("final_hit_same_track_episode_dedup_ms", final_hit_same_track_episode_dedup_ms),
        ("strong_reasoning_track_project_extend_px", strong_reasoning_track_project_extend_px),
        ("strong_reasoning_track_project_max_last_dist_px", strong_reasoning_track_project_max_last_dist_px),
    ):
        text = str(raw or "").strip()
        if text:
            noise_payload[key] = float(text)
    if str(human_noise_track_start_bbox_filter_enable or "").strip() != "":
        noise_payload["human_noise_track_start_bbox_filter_enable"] = parse_bool(human_noise_track_start_bbox_filter_enable)
    policy = str(human_noise_track_start_bbox_policy or "").strip().lower()
    if policy:
        if policy not in {"soft", "hard", "off"}:
            raise SystemExit(f"unsupported human noise track start bbox policy: {policy}")
        noise_payload["human_noise_track_start_bbox_policy"] = policy
    mode = str(projected_mask_final_mode or "").strip().lower()
    if mode:
        if mode not in {"pending_only", "shadow_only", "final_allowed"}:
            raise SystemExit(f"unsupported projected mask final mode: {mode}")
        noise_payload["projected_mask_final_mode"] = mode
    if str(projected_mask_require_terminal_or_turn or "").strip() != "":
        noise_payload["projected_mask_require_terminal_or_turn"] = parse_bool(projected_mask_require_terminal_or_turn)
    if str(final_hit_same_track_episode_dedup_enable or "").strip() != "":
        noise_payload["final_hit_same_track_episode_dedup_enable"] = parse_bool(final_hit_same_track_episode_dedup_enable)
    if str(strong_reasoning_track_project_enable or "").strip() != "":
        noise_payload["strong_reasoning_track_project_enable"] = parse_bool(strong_reasoning_track_project_enable)
    if noise_payload:
        hit_path = sync / "hit_judge_control.json"
        hit = _merge_control_json(hit_path, noise_payload, source)
        changed["hit_judge_control"] = hit

    raw_cmd = str(raw_record_cmd or "").strip().lower()
    if raw_cmd:
        if raw_cmd not in {"start", "stop", "toggle"}:
            raise SystemExit(f"unsupported raw record command: {raw_cmd}")
        raw_targets = [
            x.strip()
            for x in str(raw_record_targets or "event_raw,mvs_raw").replace(";", ",").split(",")
            if x.strip()
        ]
        if not raw_targets:
            raw_targets = ["event_raw", "mvs_raw"]
        raw_sid = str(raw_record_session_id or "").strip()
        if raw_cmd == "start" and not raw_sid:
            raw_sid = time.strftime("%Y%m%d_%H%M%S_full_remote")
        raw_payload: Dict[str, Any] = {
            "cmd": raw_cmd,
            "targets": raw_targets,
            "session_id": raw_sid,
            "wall_us": now_wall_us(),
        }
        raw = _merge_control_json(sync / "record_control.json", raw_payload, source)
        changed["record_control"] = raw

    pds_cmd = str(person_dataset_cmd or "").strip().lower()
    if pds_cmd:
        if pds_cmd not in {"start", "stop"}:
            raise SystemExit(f"unsupported person dataset command: {pds_cmd}")
        dtype = str(person_dataset_type or "mixed").strip().lower()
        if dtype not in {"negative", "positive", "mixed", "battle_1v1"}:
            dtype = "mixed"
        kind = str(person_dataset_kind or "").strip().lower()
        if kind in {"", "person", "person_id"}:
            kind = "person_id"
        elif kind in {"evidence", "evidence_replay", "evidence_replay_dataset"}:
            kind = "evidence_replay"
        else:
            raise SystemExit(f"unsupported person dataset kind: {kind}")
        pds_path = sync / "person_id_dataset_control.json"
        payload: Dict[str, Any] = {
            "cmd": pds_cmd,
            "dataset_kind": kind,
            "dataset_type": dtype,
            "label": str(person_dataset_label or f"{dtype}_{source}"),
            "description": f"{source}_remote_ops",
        }
        if str(person_dataset_session_id or "").strip():
            payload["session_id"] = str(person_dataset_session_id).strip()
        if str(person_dataset_include_mask_polygons or "").strip() != "":
            payload["include_mask_polygons"] = parse_bool(person_dataset_include_mask_polygons)
        elif kind == "evidence_replay":
            payload["include_mask_polygons"] = True
        if str(person_dataset_include_bullet_trajectory or "").strip() != "":
            payload["include_bullet_trajectory"] = parse_bool(person_dataset_include_bullet_trajectory)
        elif kind == "evidence_replay":
            payload["include_bullet_trajectory"] = True
        if str(person_dataset_include_hit_debug or "").strip() != "":
            payload["include_hit_debug"] = parse_bool(person_dataset_include_hit_debug)
        elif kind == "evidence_replay":
            payload["include_hit_debug"] = True
        pds = _merge_control_json(pds_path, payload, source)
        changed["person_id_dataset_control"] = pds

    gpid_payload: Dict[str, Any] = {}
    backend = str(global_person_filter_backend or "").strip().lower()
    if backend:
        if backend not in {"alpha_beta", "kalman_cv"}:
            raise SystemExit(f"unsupported global person filter backend: {backend}")
        gpid_payload["filter_backend"] = backend
    if bool(global_person_reset_tracks):
        gpid_payload["reset_tracks"] = True
        gpid_payload["reset_reason"] = source

    float_fields = {
        "cluster_gate_m": global_person_cluster_gate_m,
        "match_gate_m": global_person_match_gate_m,
        "lost_match_gate_m": global_person_lost_match_gate_m,
        "reid_memory_ms": global_person_reid_memory_ms,
        "retire_after_ms": global_person_retire_after_ms,
        "duplicate_suppress_dist_m": global_person_duplicate_suppress_dist_m,
        "birth_match_gate_m": global_person_birth_match_gate_m,
        "birth_near_track_suppress_m": global_person_birth_near_track_suppress_m,
        "birth_candidate_ttl_ms": global_person_birth_candidate_ttl_ms,
        "kalman_process_noise_m": global_person_kalman_process_noise_m,
        "kalman_measurement_noise_m": global_person_kalman_measurement_noise_m,
    }
    for key, raw in float_fields.items():
        if str(raw or "").strip() != "":
            gpid_payload[key] = float(raw)
    if str(global_person_birth_confirm_hits or "").strip() != "":
        gpid_payload["birth_confirm_hits"] = int(global_person_birth_confirm_hits)
    if str(global_person_birth_quarantine_enable or "").strip() != "":
        gpid_payload["birth_quarantine_enable"] = parse_bool(str(global_person_birth_quarantine_enable))
    if str(global_person_kalman_use_for_association or "").strip() != "":
        gpid_payload["kalman_use_for_association"] = parse_bool(str(global_person_kalman_use_for_association))

    if gpid_payload:
        target = str(global_person_target or "shadow").strip().lower()
        if target not in {"official", "shadow", "both"}:
            raise SystemExit(f"unsupported global person target: {target}")
        targets = ["official", "shadow"] if target == "both" else [target]
        for item in targets:
            path = sync / ("global_person_id_control.json" if item == "official" else "global_person_shadow_control.json")
            ctrl = _merge_control_json(path, gpid_payload, source)
            changed[f"global_person_id_{item}_control"] = ctrl

    return changed


def summarize(project: Path, run_dir: Path) -> Dict[str, Any]:
    rc_path = run_dir / "launcher_rc.txt"
    rc = read_text(rc_path, max_chars=200).strip() if rc_path.exists() else ""
    before = parse_mask_list(run_dir / "event_human_mask_stats_before.txt")
    after = parse_mask_list(run_dir / "event_human_mask_stats_after.txt")
    mask_growth = []
    for name in sorted(set(before) | set(after)):
        b = before.get(name)
        a = after.get(name)
        if b is None:
            mask_growth.append({"name": name, "state": "new", "after": a})
        elif a is None:
            mask_growth.append({"name": name, "state": "missing_after", "before": b})
        elif a[0] != b[0] or abs(a[1] - b[1]) > 0.0001:
            mask_growth.append({"name": name, "state": "changed", "before": b, "after": a})

    snapshots = []
    snapshot_paths = sorted(
        (run_dir / "control_checks").glob("*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    for path in snapshot_paths:
        snapshots.append(read_json(path))

    latest = max(snapshots, key=lambda s: float(s.get("ts", 0.0) or 0.0)) if snapshots else {}
    runtime_snapshots = [s for s in snapshots if str(s.get("tag", "")).lower() not in {"final", "after_stop"}]
    latest_runtime = max(runtime_snapshots, key=lambda s: float(s.get("ts", 0.0) or 0.0)) if runtime_snapshots else latest
    audit_source = latest_runtime if latest_runtime else latest
    runtime_system = latest_runtime.get("system_status", {}) if isinstance(latest_runtime, dict) else {}
    runtime_summary = runtime_system.get("summary", {}) if isinstance(runtime_system, dict) else {}
    if not isinstance(runtime_summary, dict):
        runtime_summary = {}
    start_epoch_s = to_int(read_text(run_dir / "start_epoch.txt", max_chars=200).strip(), 0)
    end_epoch_s = to_int(read_text(run_dir / "end_epoch.txt", max_chars=200).strip(), 0)
    start_wall_us = int(start_epoch_s * 1_000_000) if start_epoch_s > 0 else 0
    end_wall_us = int((end_epoch_s + 5) * 1_000_000) if end_epoch_s > 0 else 0
    archived_overview = read_json(run_dir / "status" / "event_overlay_overview_status.json")
    archived_overview_snapshot = {
        "tag": "archived_status",
        "event_overlay_overview": {
            "age_ms": archived_overview.get("_age_ms"),
            "event_overlay": compact_event_overlay_rows(archived_overview),
        },
    }
    runtime_summary, overview_reconciled_links = reconcile_runtime_summary_with_overview(
        runtime_summary,
        latest_runtime,
        latest,
        archived_overview_snapshot,
    )
    reference_wall_us = int(float(latest_runtime.get("ts", latest.get("ts", 0.0)) or 0.0) * 1_000_000)
    runtime_summary, worker_reconciled_links = reconcile_runtime_summary_with_worker_sidecars(
        runtime_summary,
        run_dir,
        latest_runtime,
        latest,
        archived_overview_snapshot,
        reference_wall_us=reference_wall_us,
        min_wall_us=start_wall_us,
        max_wall_us=end_wall_us,
    )
    runtime_bev = latest_runtime.get("global_bev", {}) if isinstance(latest_runtime, dict) and isinstance(latest_runtime.get("global_bev"), dict) else {}
    bev_published_us = to_int(runtime_bev.get("published_wall_us"), 0)
    alignment = runtime_bev.get("alignment", {}) if isinstance(runtime_bev.get("alignment"), dict) else {}
    event_worker_sync = latest_runtime.get("event_worker_sync", {}) if isinstance(latest_runtime, dict) and isinstance(latest_runtime.get("event_worker_sync"), dict) else {}
    event_display_sync = latest_runtime.get("event_display_sync", {}) if isinstance(latest_runtime, dict) and isinstance(latest_runtime.get("event_display_sync"), dict) else {}
    if not isinstance(event_worker_sync.get("cameras"), dict) or not event_worker_sync.get("cameras"):
        archived_status_dir = run_dir / "status"
        archived_bev = read_json(archived_status_dir / "global_bev_status.json")
        if not archived_bev.get("_error"):
            event_worker_sync = collect_event_worker_sync_diagnostics(archived_status_dir, archived_bev)
            if not event_display_sync:
                event_display_sync = collect_bev_event_display_sync_diagnostics(archived_bev)
    status_freshness = {
        "start_wall_us": int(start_wall_us),
        "global_bev_published_wall_us": int(bev_published_us),
        "global_bev_from_this_run": bool(start_wall_us > 0 and bev_published_us >= start_wall_us),
        "global_bev_alignment_present": bool(isinstance(alignment, dict) and alignment),
        "global_bev_alignment_mode": alignment.get("mode") if isinstance(alignment, dict) else None,
        "global_bev_alignment_state": alignment.get("state") if isinstance(alignment, dict) else None,
    }
    source_mode_request = read_json(run_dir / "source_mode_request.json")
    source_mode_effective = read_json(run_dir / "status" / "source_mode_effective.json")
    if source_mode_effective.get("_error"):
        source_mode_effective = read_json(run_dir / "status" / "source_mode_effective_runtime.json")
    if source_mode_effective.get("_error"):
        source_mode_effective = read_json(run_dir / "source_mode_effective.json")
    v25_resolved_manifest = read_json(run_dir / "status" / "v25_resolved_manifest_latest.json")
    if v25_resolved_manifest.get("_error"):
        v25_resolved_manifest = read_json(run_dir / "status" / "v25_resolved_manifest_runtime.json")
    if v25_resolved_manifest.get("_error"):
        v25_resolved_manifest = read_json(run_dir / "v25_resolved_manifest_latest.json")
    event_broker_stats = summarize_event_broker_stats(run_dir)
    mvs_file_broker_stats = summarize_mvs_file_broker_stats(run_dir, source_mode_request)
    runner_tail = read_text(run_dir / "runner.out", max_chars=12000)
    report = {
        "kind": "fullsys_test_report",
        "ts": time.time(),
        "project": str(project),
        "run_dir": str(run_dir),
        "launcher_rc": rc,
        "launcher_timeout_ok": rc == "124",
        "launcher_clean_exit": rc in {"0", "124"},
        "mask_stats_jsonl_no_growth": len(mask_growth) == 0,
        "mask_stats_jsonl_changes": mask_growth,
        "snapshot_count": len(snapshots),
        "latest_snapshot": latest,
        "latest_runtime_snapshot": latest_runtime,
        "runtime_overall_state": runtime_summary.get("overall_state"),
        "runtime_broken_links": runtime_summary.get("broken_links", []),
        "runtime_degraded_links": runtime_summary.get("degraded_links", []),
        "runtime_overview_reconciled_links": overview_reconciled_links,
        "runtime_worker_reconciled_links": worker_reconciled_links,
        "source_mode_request": source_mode_request,
        "source_mode_effective": source_mode_effective,
        "v25_resolved_manifest": v25_resolved_manifest,
        "event_broker_stats": event_broker_stats,
        "mvs_file_broker_stats": mvs_file_broker_stats,
        "event_worker_sync": event_worker_sync,
        "event_display_sync": event_display_sync,
        "status_freshness": status_freshness,
        "status_key_audit": audit_source.get("status_key_audit", {}) if isinstance(audit_source, dict) else {},
        "final_snapshot_is_after_shutdown": str(latest.get("tag", "")).lower() == "final" if isinstance(latest, dict) else False,
        "runner_tail": runner_tail,
        "archive": str(run_dir) + ".tar.gz",
    }
    write_json(run_dir / "test_report.json", report)
    return report


def compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    runtime = report.get("latest_runtime_snapshot", {})
    if not isinstance(runtime, dict) or not runtime:
        runtime = report.get("latest_snapshot", {})
    if not isinstance(runtime, dict):
        runtime = {}
    global_bev = runtime.get("global_bev", {}) if isinstance(runtime.get("global_bev"), dict) else {}
    global_yolo = runtime.get("global_yolo", {}) if isinstance(runtime.get("global_yolo"), dict) else {}
    global_person = runtime.get("global_person", {}) if isinstance(runtime.get("global_person"), dict) else {}
    if not global_person and isinstance(global_bev.get("global_person"), dict):
        global_person = global_bev.get("global_person", {})
    global_person_status = runtime.get("global_person_id_status", {}) if isinstance(runtime.get("global_person_id_status"), dict) else {}
    global_person_shadow_status = runtime.get("global_person_id_shadow_status", {}) if isinstance(runtime.get("global_person_id_shadow_status"), dict) else {}
    person_dataset = runtime.get("person_id_dataset", {}) if isinstance(runtime.get("person_id_dataset"), dict) else {}
    experiment = runtime.get("experiment", {}) if isinstance(runtime.get("experiment"), dict) else {}
    status_audit = report.get("status_key_audit", {}) if isinstance(report.get("status_key_audit"), dict) else {}
    alignment = global_bev.get("alignment", {}) if isinstance(global_bev.get("alignment"), dict) else {}
    align_quality = alignment.get("quality", {}) if isinstance(alignment.get("quality"), dict) else {}
    layer = global_bev.get("event_layer", {}) if isinstance(global_bev.get("event_layer"), dict) else {}
    layer_presentation = layer.get("presentation_worker", {}) if isinstance(layer.get("presentation_worker"), dict) else {}
    event_worker_sync = report.get("event_worker_sync", {}) if isinstance(report.get("event_worker_sync"), dict) else {}
    if not event_worker_sync and isinstance(runtime.get("event_worker_sync"), dict):
        event_worker_sync = runtime.get("event_worker_sync", {})
    event_display_sync = report.get("event_display_sync", {}) if isinstance(report.get("event_display_sync"), dict) else {}
    if not event_display_sync and isinstance(runtime.get("event_display_sync"), dict):
        event_display_sync = runtime.get("event_display_sync", {})
    source_mode_effective = report.get("source_mode_effective", {}) if isinstance(report.get("source_mode_effective"), dict) else {}
    v25_resolved_manifest = report.get("v25_resolved_manifest", {}) if isinstance(report.get("v25_resolved_manifest"), dict) else {}
    return {
        "kind": "fullsys_test_report_compact",
        "run_dir": report.get("run_dir"),
        "archive": report.get("archive"),
        "launcher_rc": report.get("launcher_rc"),
        "launcher_timeout_ok": report.get("launcher_timeout_ok"),
        "launcher_clean_exit": report.get("launcher_clean_exit"),
        "runtime_overall_state": report.get("runtime_overall_state"),
        "runtime_broken_links": report.get("runtime_broken_links", []),
        "runtime_degraded_links": report.get("runtime_degraded_links", []),
        "runtime_overview_reconciled_links": report.get("runtime_overview_reconciled_links", []),
        "runtime_worker_reconciled_links": report.get("runtime_worker_reconciled_links", []),
        "source_mode_request": {
            "profile": (report.get("source_mode_request", {}) if isinstance(report.get("source_mode_request"), dict) else {}).get("profile"),
            "launch_extra_args": (report.get("source_mode_request", {}) if isinstance(report.get("source_mode_request"), dict) else {}).get("launch_extra_args"),
            "dataset_dir": (report.get("source_mode_request", {}) if isinstance(report.get("source_mode_request"), dict) else {}).get("dataset_dir"),
            "mvs_source_mode_requested": (report.get("source_mode_request", {}) if isinstance(report.get("source_mode_request"), dict) else {}).get("mvs_source_mode_requested"),
        },
        "source_mode_effective": {
            "event_source_mode_effective": source_mode_effective.get("event_source_mode_effective"),
            "mvs_source_mode_effective": source_mode_effective.get("mvs_source_mode_effective"),
            "dataset_dir": source_mode_effective.get("dataset_dir"),
            "dataset_id": source_mode_effective.get("dataset_id"),
            "session_id": source_mode_effective.get("session_id"),
            "event_file_fifo_broker_enabled": source_mode_effective.get("event_file_fifo_broker_enabled"),
            "event_live_hal_enabled": source_mode_effective.get("event_live_hal_enabled"),
            "mvs_file_frame_broker_enabled": source_mode_effective.get("mvs_file_frame_broker_enabled"),
            "mvs_live_camera_enabled": source_mode_effective.get("mvs_live_camera_enabled"),
            "event_file_fifo_broker_raw_camera_count": source_mode_effective.get("event_file_fifo_broker_raw_camera_count"),
            "mvs_file_frame_broker_dataset_dir": source_mode_effective.get("mvs_file_frame_broker_dataset_dir"),
            "capabilities": source_mode_effective.get("capabilities"),
            "mutual_exclusion": source_mode_effective.get("mutual_exclusion"),
        },
        "v25_resolved_manifest": {
            "dataset_id": v25_resolved_manifest.get("dataset_id"),
            "session_id": v25_resolved_manifest.get("session_id"),
            "scenario": v25_resolved_manifest.get("scenario"),
            "capabilities": v25_resolved_manifest.get("capabilities"),
            "degraded_reasons": v25_resolved_manifest.get("degraded_reasons"),
            "recommended_launch": v25_resolved_manifest.get("recommended_launch"),
        },
        "event_broker_stats": report.get("event_broker_stats", {}),
        "mvs_file_broker_stats": report.get("mvs_file_broker_stats", {}),
        "event_worker_sync": {
            "diagnostic_scope": event_worker_sync.get("diagnostic_scope"),
            "display_sync_note": event_worker_sync.get("display_sync_note"),
            "bev_target_sync_id": event_worker_sync.get("bev_target_sync_id"),
            "locked_cam_count": event_worker_sync.get("locked_cam_count"),
            "direct_index_cam_count": event_worker_sync.get("direct_index_cam_count"),
            "total_selected_edge_count": event_worker_sync.get("total_selected_edge_count"),
            "mapped_sync_min": event_worker_sync.get("mapped_sync_min"),
            "mapped_sync_max": event_worker_sync.get("mapped_sync_max"),
            "mapped_sync_span": event_worker_sync.get("mapped_sync_span"),
            "mapped_sync_delta_min_to_bev": event_worker_sync.get("mapped_sync_delta_min_to_bev"),
            "mapped_sync_delta_max_to_bev": event_worker_sync.get("mapped_sync_delta_max_to_bev"),
            "cameras": event_worker_sync.get("cameras", {}),
        },
        "event_display_sync": {
            "diagnostic_scope": event_display_sync.get("diagnostic_scope"),
            "state": event_display_sync.get("state"),
            "synced": event_display_sync.get("synced"),
            "enabled": event_display_sync.get("enabled"),
            "mode": event_display_sync.get("mode"),
            "presentation_state": event_display_sync.get("presentation_state"),
            "worker_state": event_display_sync.get("worker_state"),
            "target_sync_id": event_display_sync.get("target_sync_id"),
            "alignment_overall_state": event_display_sync.get("alignment_overall_state"),
            "alignment_event_synced_cam_count": event_display_sync.get("alignment_event_synced_cam_count"),
            "bundle_lookup_state": event_display_sync.get("bundle_lookup_state"),
            "bundle_target_delta_ticks": event_display_sync.get("bundle_target_delta_ticks"),
            "compare_lookup_state": event_display_sync.get("compare_lookup_state"),
            "compare_target_delta_ticks": event_display_sync.get("compare_target_delta_ticks"),
            "compare_tolerance_ticks": event_display_sync.get("compare_tolerance_ticks"),
            "display_compare_state": event_display_sync.get("display_compare_state"),
            "display_mismatch_cam_count": event_display_sync.get("display_mismatch_cam_count"),
            "bundle_hit_ratio": event_display_sync.get("bundle_hit_ratio"),
            "bundle_hit_count": event_display_sync.get("bundle_hit_count"),
            "bundle_miss_count": event_display_sync.get("bundle_miss_count"),
            "compare_hit_ratio": event_display_sync.get("compare_hit_ratio"),
            "consume_exact_count": event_display_sync.get("consume_exact_count"),
            "consume_near_count": event_display_sync.get("consume_near_count"),
            "active_consume_count": event_display_sync.get("active_consume_count"),
            "fallback_count": event_display_sync.get("fallback_count"),
            "cached_target_min": event_display_sync.get("cached_target_min"),
            "cached_target_max": event_display_sync.get("cached_target_max"),
            "updated_age_ms": event_display_sync.get("updated_age_ms"),
        },
        "status_freshness": report.get("status_freshness", {}),
        "status_key_ok": status_audit.get("ok"),
        "mask_stats_jsonl_no_growth": report.get("mask_stats_jsonl_no_growth"),
        "final_snapshot_is_after_shutdown": report.get("final_snapshot_is_after_shutdown"),
        "global_bev": {
            "published_wall_us": global_bev.get("published_wall_us"),
            "render_fps": global_bev.get("render_fps"),
            "output_fps": global_bev.get("output_fps"),
            "target_sync_id": global_bev.get("target_sync_id"),
            "event_layer_dilate_backend": layer.get("dilate_backend"),
            "event_layer_dilate_px": layer.get("dilate_px"),
            "mvs_texture_upload": global_bev.get("mvs_texture_upload"),
            "event_layer_display_budget_ms": layer.get("display_budget_ms"),
            "event_layer_display_budget_elapsed_ms": layer.get("display_budget_elapsed_ms"),
            "event_layer_display_budget_held_cam_count": layer.get("display_budget_held_cam_count"),
            "event_layer_display_budget_overrun_rate": layer.get("display_budget_overrun_rate"),
            "event_layer_display_budget_order": layer.get("display_budget_order"),
            "event_layer_presentation_worker": {
                "enabled": layer_presentation.get("enabled"),
                "mode": layer_presentation.get("mode"),
                "state": layer_presentation.get("state"),
                "worker_state": layer_presentation.get("worker_state"),
                "payload_policy": layer_presentation.get("payload_policy"),
                "read_points": layer_presentation.get("read_points"),
                "compare_state": layer_presentation.get("compare_state"),
                "display_compare_state": layer_presentation.get("display_compare_state"),
                "delta_compare_state": layer_presentation.get("delta_compare_state"),
                "mismatch_cam_count": layer_presentation.get("mismatch_cam_count"),
                "display_mismatch_cam_count": layer_presentation.get("display_mismatch_cam_count"),
                "delta_mismatch_cam_count": layer_presentation.get("delta_mismatch_cam_count"),
                "compare_lookup_state": layer_presentation.get("compare_lookup_state"),
                "compare_target_delta_ticks": layer_presentation.get("compare_target_delta_ticks"),
                "compare_tolerance_ticks": layer_presentation.get("compare_tolerance_ticks"),
                "compare_exact_count": layer_presentation.get("compare_exact_count"),
                "compare_near_count": layer_presentation.get("compare_near_count"),
                "compare_miss_count": layer_presentation.get("compare_miss_count"),
                "compare_hit_ratio": layer_presentation.get("compare_hit_ratio"),
                "target_delta_ticks": layer_presentation.get("target_delta_ticks"),
                "updated_age_ms": layer_presentation.get("updated_age_ms"),
                "build_elapsed_ms": layer_presentation.get("build_elapsed_ms"),
                "build_p95_ms": layer_presentation.get("build_p95_ms"),
                "active_consume_elapsed_ms": layer_presentation.get("active_consume_elapsed_ms"),
                "active_consume_count": layer_presentation.get("active_consume_count"),
                "fallback_count": layer_presentation.get("fallback_count"),
                "fallback_reason": layer_presentation.get("fallback_reason"),
                "last_fallback_reason": layer_presentation.get("last_fallback_reason"),
                "bundle_lookup_state": layer_presentation.get("bundle_lookup_state"),
                "bundle_target_delta_ticks": layer_presentation.get("bundle_target_delta_ticks"),
                "bundle_hit_count": layer_presentation.get("bundle_hit_count"),
                "bundle_miss_count": layer_presentation.get("bundle_miss_count"),
                "bundle_hit_ratio": layer_presentation.get("bundle_hit_ratio"),
                "consume_exact_count": layer_presentation.get("consume_exact_count"),
                "consume_near_count": layer_presentation.get("consume_near_count"),
                "bundle_ring_size": layer_presentation.get("bundle_ring_size"),
                "bundle_ring_count": layer_presentation.get("bundle_ring_count"),
                "cached_target_min": layer_presentation.get("cached_target_min"),
                "cached_target_max": layer_presentation.get("cached_target_max"),
            },
            "alignment": {
                "mode": alignment.get("mode"),
                "state": alignment.get("state"),
                "overall_state": alignment.get("overall_state"),
                "target_sync_id": alignment.get("target_sync_id"),
                "delay_ms": alignment.get("delay_ms"),
                "mvs_synced_cam_count": align_quality.get("mvs_synced_cam_count"),
                "mvs_missing_history_cam_count": align_quality.get("mvs_missing_history_cam_count"),
                "mvs_metadata_outlier_cam_count": align_quality.get("mvs_metadata_outlier_cam_count"),
                "event_synced_cam_count": align_quality.get("event_synced_cam_count"),
                "event_sync_outlier_cam_count": align_quality.get("event_sync_outlier_cam_count"),
                "worker_elapsed_p95_ms": alignment.get("worker_elapsed_p95_ms"),
                "gui_refresh_p95_ms": alignment.get("gui_refresh_p95_ms"),
                "gui_upload_selected_p95_ms": alignment.get("gui_upload_selected_p95_ms"),
                "gui_upload_event_layer_p95_ms": alignment.get("gui_upload_event_layer_p95_ms"),
                "gui_paint_total_p95_ms": alignment.get("gui_paint_total_p95_ms"),
                "mvs_upload_budget_ms": alignment.get("mvs_upload_budget_ms"),
                "mvs_upload_elapsed_ms": alignment.get("mvs_upload_elapsed_ms"),
                "mvs_upload_held_by_budget_cam_count": alignment.get("mvs_upload_held_by_budget_cam_count"),
                "mvs_upload_budget_overrun_rate": alignment.get("mvs_upload_budget_overrun_rate"),
                "event_layer_display_budget_ms": alignment.get("event_layer_display_budget_ms"),
                "event_layer_display_budget_elapsed_ms": alignment.get("event_layer_display_budget_elapsed_ms"),
                "event_layer_display_budget_held_cam_count": alignment.get("event_layer_display_budget_held_cam_count"),
                "event_layer_display_budget_overrun_rate": alignment.get("event_layer_display_budget_overrun_rate"),
                "gui_event_sparse_read_p95_ms": alignment.get("gui_event_sparse_read_p95_ms"),
                "gui_event_sparse_sync_gate_p95_ms": alignment.get("gui_event_sparse_sync_gate_p95_ms"),
                "gui_event_sparse_project_p95_ms": alignment.get("gui_event_sparse_project_p95_ms"),
                "mvs_offset_ms_by_cam": alignment.get("mvs_offset_ms_by_cam"),
                "event_delta_ms_by_cam": alignment.get("event_delta_ms_by_cam"),
            },
        },
        "global_yolo": {
            "worker_count": global_yolo.get("worker_count"),
            "fps": global_yolo.get("fps"),
            "inferred_fps_total": global_yolo.get("inferred_fps_total"),
            "published_fps_total": global_yolo.get("published_fps_total"),
            "async_record_postprocess_enabled": global_yolo.get("async_record_postprocess_enabled"),
            "postprocess_queue_depth": global_yolo.get("postprocess_queue_depth"),
            "postprocess_lag_ms": global_yolo.get("postprocess_lag_ms"),
            "postprocess_backpressure_ms": global_yolo.get("postprocess_backpressure_ms"),
            "postprocess_dropped_batches": global_yolo.get("postprocess_dropped_batches"),
            "polygon_contract_ok": global_yolo.get("polygon_contract_ok"),
            "published_records_with_polygon_ratio": global_yolo.get("published_records_with_polygon_ratio"),
            "published_persons_with_polygon_ratio": global_yolo.get("published_persons_with_polygon_ratio"),
            "input_alignment_audit": global_yolo.get("input_alignment_audit"),
            "input_buffer": global_yolo.get("input_buffer"),
            "summary": global_yolo.get("summary"),
        },
        "global_person": {
            "track_count": global_person.get("track_count"),
            "tracks_len": len(global_person.get("tracks", [])) if isinstance(global_person.get("tracks"), list) else None,
        },
        "global_person_id_status": {
            "track_count": global_person_status.get("track_count"),
            "filter_backend": global_person_status.get("filter_backend"),
            "stream_role": global_person_status.get("stream_role"),
            "control_seq": (global_person_status.get("control", {}) if isinstance(global_person_status.get("control"), dict) else {}).get("seq"),
            "created_tracks": (global_person_status.get("stats", {}) if isinstance(global_person_status.get("stats"), dict) else {}).get("created_tracks"),
            "id_switch_suspects": (global_person_status.get("stats", {}) if isinstance(global_person_status.get("stats"), dict) else {}).get("id_switch_suspects"),
            "birth_candidate_count": global_person_status.get("birth_candidate_count"),
            "birth_promoted": (global_person_status.get("stats", {}) if isinstance(global_person_status.get("stats"), dict) else {}).get("birth_candidates_promoted"),
            "birth_suppressed_near_track": (global_person_status.get("stats", {}) if isinstance(global_person_status.get("stats"), dict) else {}).get("birth_suppressed_near_track"),
        },
        "global_person_id_shadow_status": {
            "track_count": global_person_shadow_status.get("track_count"),
            "filter_backend": global_person_shadow_status.get("filter_backend"),
            "stream_role": global_person_shadow_status.get("stream_role"),
            "control_seq": (global_person_shadow_status.get("control", {}) if isinstance(global_person_shadow_status.get("control"), dict) else {}).get("seq"),
            "created_tracks": (global_person_shadow_status.get("stats", {}) if isinstance(global_person_shadow_status.get("stats"), dict) else {}).get("created_tracks"),
            "id_switch_suspects": (global_person_shadow_status.get("stats", {}) if isinstance(global_person_shadow_status.get("stats"), dict) else {}).get("id_switch_suspects"),
            "birth_candidate_count": global_person_shadow_status.get("birth_candidate_count"),
            "birth_promoted": (global_person_shadow_status.get("stats", {}) if isinstance(global_person_shadow_status.get("stats"), dict) else {}).get("birth_candidates_promoted"),
            "birth_suppressed_near_track": (global_person_shadow_status.get("stats", {}) if isinstance(global_person_shadow_status.get("stats"), dict) else {}).get("birth_suppressed_near_track"),
        },
        "person_id_dataset": {
            "state": person_dataset.get("state"),
            "recording": person_dataset.get("recording"),
            "dataset_kind": person_dataset.get("dataset_kind"),
            "dataset_type": person_dataset.get("dataset_type"),
            "session_id": person_dataset.get("session_id"),
            "human_rows_total": (person_dataset.get("counts", {}) if isinstance(person_dataset.get("counts"), dict) else {}).get("human_rows_total"),
            "human_person_rows_total": (person_dataset.get("counts", {}) if isinstance(person_dataset.get("counts"), dict) else {}).get("human_person_rows_total"),
            "bullet_point_rows": (person_dataset.get("counts", {}) if isinstance(person_dataset.get("counts"), dict) else {}).get("bullet_point_rows"),
            "pseudo_hit_rows": (person_dataset.get("counts", {}) if isinstance(person_dataset.get("counts"), dict) else {}).get("pseudo_hit_rows"),
            "bad_lines": (person_dataset.get("counts", {}) if isinstance(person_dataset.get("counts"), dict) else {}).get("bad_lines"),
        },
        "experiment": {
            "experiment_id": experiment.get("experiment_id"),
            "stage": experiment.get("stage"),
            "config_hash": experiment.get("config_hash"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Remote full-system test status/control helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot")
    snap.add_argument("--project", default=".")
    snap.add_argument("--run-dir", default="")
    snap.add_argument("--tag", default="snapshot")

    live = sub.add_parser("live-status")
    live.add_argument("--project", default=".")
    live.add_argument("--run-dir", default="")
    live.add_argument("--label", default="")
    live.add_argument("--tag", default="live")
    live.add_argument("--write-latest", action="store_true")
    live.add_argument("--compact", action="store_true")

    ctrl = sub.add_parser("set-control")
    ctrl.add_argument("--project", default=".")
    ctrl.add_argument("--overview-visible", choices=["true", "false", ""], default="")
    ctrl.add_argument("--bev-dilate-px", default="")
    ctrl.add_argument("--show-event-layer", choices=["true", "false", ""], default="")
    ctrl.add_argument("--hit-enabled", choices=["true", "false", ""], default="")
    ctrl.add_argument("--final-hit-authority", choices=["strong_gate", "strong_reasoning", "both_shadow_compare", ""], default="")
    ctrl.add_argument("--strong-gate-enabled", choices=["true", "false", ""], default="")
    ctrl.add_argument("--strong-reasoning-enabled", choices=["true", "false", ""], default="")
    ctrl.add_argument("--strong-reasoning-shadow-only", choices=["true", "false", ""], default="")
    ctrl.add_argument("--human-noise-filter-enable", choices=["true", "false", ""], default="")
    ctrl.add_argument("--human-noise-filter-mode", choices=["off", "shadow", "soft", "hard", ""], default="")
    ctrl.add_argument("--human-noise-soft-threshold", default="")
    ctrl.add_argument("--human-noise-hard-threshold", default="")
    ctrl.add_argument("--human-noise-track-start-bbox-filter-enable", choices=["true", "false", ""], default="")
    ctrl.add_argument("--human-noise-track-start-bbox-expand-px", default="")
    ctrl.add_argument("--human-noise-track-start-bbox-policy", choices=["soft", "hard", "off", ""], default="")
    ctrl.add_argument("--final-hit-cross-camera-dedup-ms", default="")
    ctrl.add_argument("--projected-mask-final-mode", choices=["pending_only", "shadow_only", "final_allowed", ""], default="")
    ctrl.add_argument("--projected-mask-require-terminal-or-turn", choices=["true", "false", ""], default="")
    ctrl.add_argument("--same-track-episode-dedup-enable", choices=["true", "false", ""], default="")
    ctrl.add_argument("--same-track-episode-dedup-ms", default="")
    ctrl.add_argument("--strong-reasoning-track-project-enable", choices=["true", "false", ""], default="")
    ctrl.add_argument("--strong-reasoning-track-project-extend-px", default="")
    ctrl.add_argument("--strong-reasoning-track-project-max-last-dist-px", default="")
    ctrl.add_argument("--raw-record-cmd", choices=["start", "stop", "toggle", ""], default="")
    ctrl.add_argument("--raw-record-targets", default="event_raw,mvs_raw")
    ctrl.add_argument("--raw-record-session-id", default="")
    ctrl.add_argument("--person-dataset-cmd", choices=["start", "stop", ""], default="")
    ctrl.add_argument("--person-dataset-kind", choices=["person_id", "evidence_replay", ""], default="")
    ctrl.add_argument("--person-dataset-type", choices=["negative", "positive", "mixed", "battle_1v1", ""], default="")
    ctrl.add_argument("--person-dataset-label", default="")
    ctrl.add_argument("--person-dataset-session-id", default="")
    ctrl.add_argument("--person-dataset-include-mask-polygons", choices=["true", "false", ""], default="")
    ctrl.add_argument("--person-dataset-include-bullet-trajectory", choices=["true", "false", ""], default="")
    ctrl.add_argument("--person-dataset-include-hit-debug", choices=["true", "false", ""], default="")
    ctrl.add_argument("--global-person-target", choices=["official", "shadow", "both", ""], default="")
    ctrl.add_argument("--global-person-filter-backend", choices=["alpha_beta", "kalman_cv", ""], default="")
    ctrl.add_argument("--global-person-reset-tracks", action="store_true")
    ctrl.add_argument("--global-person-cluster-gate-m", default="")
    ctrl.add_argument("--global-person-match-gate-m", default="")
    ctrl.add_argument("--global-person-lost-match-gate-m", default="")
    ctrl.add_argument("--global-person-reid-memory-ms", default="")
    ctrl.add_argument("--global-person-retire-after-ms", default="")
    ctrl.add_argument("--global-person-duplicate-suppress-dist-m", default="")
    ctrl.add_argument("--global-person-birth-confirm-hits", default="")
    ctrl.add_argument("--global-person-birth-match-gate-m", default="")
    ctrl.add_argument("--global-person-birth-near-track-suppress-m", default="")
    ctrl.add_argument("--global-person-birth-candidate-ttl-ms", default="")
    ctrl.add_argument("--global-person-birth-quarantine-enable", choices=["true", "false", ""], default="")
    ctrl.add_argument("--global-person-kalman-use-for-association", choices=["true", "false", ""], default="")
    ctrl.add_argument("--global-person-kalman-process-noise-m", default="")
    ctrl.add_argument("--global-person-kalman-measurement-noise-m", default="")
    ctrl.add_argument("--source", default="remote_ops")

    exp = sub.add_parser("set-experiment")
    exp.add_argument("--project", default=".")
    exp.add_argument("--experiment-id", default="")
    exp.add_argument("--stage", default="")
    exp.add_argument("--note", default="")
    exp.add_argument("--hit-profile", default="")
    exp.add_argument("--global-id-profile", default="")
    exp.add_argument("--authority-mode", choices=["strong_gate", "strong_reasoning", "both_shadow_compare", ""], default="")
    exp.add_argument("--human-noise-mode", choices=["off", "shadow", "soft", "hard", ""], default="")
    exp.add_argument("--bbox-start-expand-px", default="")
    exp.add_argument("--bbox-start-policy", choices=["soft", "hard", "off", ""], default="")
    exp.add_argument("--projected-mask-final-mode", choices=["pending_only", "shadow_only", "final_allowed", ""], default="")
    exp.add_argument("--projected-mask-require-terminal-or-turn", choices=["true", "false", ""], default="")
    exp.add_argument("--same-track-episode-dedup-ms", default="")
    exp.add_argument("--strong-reasoning-track-project-extend-px", default="")
    exp.add_argument("--strong-reasoning-track-project-max-last-dist-px", default="")
    exp.add_argument("--global-person-target", choices=["official", "shadow", "both", ""], default="")
    exp.add_argument("--global-person-filter-backend", choices=["alpha_beta", "kalman_cv", ""], default="")
    exp.add_argument("--source", default="remote_ops_experiment")

    summ = sub.add_parser("summary")
    summ.add_argument("--project", default=".")
    summ.add_argument("--run-dir", required=True)
    summ.add_argument("--compact", action="store_true")

    args = ap.parse_args()
    project = Path(args.project).resolve()

    if args.cmd == "snapshot":
        run_dir = Path(args.run_dir).resolve() if args.run_dir else None
        print(json.dumps(collect_snapshot(project, run_dir, args.tag), ensure_ascii=False))
        return 0

    if args.cmd == "live-status":
        run_dir = _resolve_run_dir(project, args.label, args.run_dir)
        status = collect_live_status(project, run_dir, args.label, args.tag)
        if args.write_latest:
            write_live_status(project, run_dir, status)
        out = status.get("compact", {}) if args.compact else status
        if args.compact:
            out = {
                "kind": status.get("kind"),
                "label": status.get("label"),
                "run": status.get("run"),
                "experiment": status.get("experiment"),
                "compact": status.get("compact"),
            }
        print(json.dumps(out, ensure_ascii=False))
        return 0

    if args.cmd == "set-control":
        overview = None if args.overview_visible == "" else parse_bool(args.overview_visible)
        dilate = None if args.bev_dilate_px == "" else float(args.bev_dilate_px)
        show = None if args.show_event_layer == "" else parse_bool(args.show_event_layer)
        hit_enabled = None if args.hit_enabled == "" else parse_bool(args.hit_enabled)
        print(json.dumps(set_control(
            project,
            overview,
            dilate,
            show,
            hit_enabled,
            args.raw_record_cmd,
            args.raw_record_targets,
            args.raw_record_session_id,
            args.person_dataset_cmd,
            args.person_dataset_type,
            args.person_dataset_label,
            args.person_dataset_session_id,
            args.source,
            args.person_dataset_kind,
            args.person_dataset_include_mask_polygons,
            args.person_dataset_include_bullet_trajectory,
            args.person_dataset_include_hit_debug,
            args.final_hit_authority,
            args.strong_gate_enabled,
            args.strong_reasoning_enabled,
            args.strong_reasoning_shadow_only,
            args.human_noise_filter_enable,
            args.human_noise_filter_mode,
            args.human_noise_soft_threshold,
            args.human_noise_hard_threshold,
            args.human_noise_track_start_bbox_filter_enable,
            args.human_noise_track_start_bbox_expand_px,
            args.human_noise_track_start_bbox_policy,
            args.final_hit_cross_camera_dedup_ms,
            args.projected_mask_final_mode,
            args.projected_mask_require_terminal_or_turn,
            args.same_track_episode_dedup_enable,
            args.same_track_episode_dedup_ms,
            args.strong_reasoning_track_project_enable,
            args.strong_reasoning_track_project_extend_px,
            args.strong_reasoning_track_project_max_last_dist_px,
            args.global_person_target,
            args.global_person_filter_backend,
            args.global_person_reset_tracks,
            args.global_person_cluster_gate_m,
            args.global_person_match_gate_m,
            args.global_person_lost_match_gate_m,
            args.global_person_reid_memory_ms,
            args.global_person_retire_after_ms,
            args.global_person_duplicate_suppress_dist_m,
            args.global_person_birth_confirm_hits,
            args.global_person_birth_match_gate_m,
            args.global_person_birth_near_track_suppress_m,
            args.global_person_birth_candidate_ttl_ms,
            args.global_person_birth_quarantine_enable,
            args.global_person_kalman_use_for_association,
            args.global_person_kalman_process_noise_m,
            args.global_person_kalman_measurement_noise_m,
        ), ensure_ascii=False))
        return 0

    if args.cmd == "set-experiment":
        print(json.dumps(set_experiment(
            project=project,
            source=args.source,
            experiment_id=args.experiment_id,
            stage=args.stage,
            note=args.note,
            hit_profile_path=args.hit_profile,
            global_id_profile_path=args.global_id_profile,
            authority_mode=args.authority_mode,
            human_noise_mode=args.human_noise_mode,
            bbox_start_expand_px=args.bbox_start_expand_px,
            bbox_start_policy=args.bbox_start_policy,
            global_person_target=args.global_person_target,
            global_person_filter_backend=args.global_person_filter_backend,
            projected_mask_final_mode=args.projected_mask_final_mode,
            projected_mask_require_terminal_or_turn=args.projected_mask_require_terminal_or_turn,
            same_track_episode_dedup_ms=args.same_track_episode_dedup_ms,
            project_extend_px=args.strong_reasoning_track_project_extend_px,
            project_max_last_dist_px=args.strong_reasoning_track_project_max_last_dist_px,
        ), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "summary":
        report = summarize(project, Path(args.run_dir).resolve())
        if args.compact:
            report = compact_report(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
