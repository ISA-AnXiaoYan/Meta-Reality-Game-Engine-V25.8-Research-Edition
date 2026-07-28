#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline compare Global Person ID filter backends on recorded datasets.

The recorded person-ID datasets contain YOLO-after-recognition rows. Most legacy
datasets strip mask data, while V19 polygon datasets preserve polygon evidence
for anchor validation. This tool replays those rows into the same
GlobalPersonIdServer core used by production, then compares the recorded
baseline with fresh official/shadow runs.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time as real_time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import global_person_id_server as gpid  # noqa: E402
import global_yolo_infer_server as gyolo  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _jsonl_iter(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        return


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _record_ts_us(rec: Dict[str, Any]) -> int:
    for key in ("capture_wall_us", "timestamp_us", "record_wall_us", "dataset_ingest_wall_us"):
        value = _safe_int(rec.get(key), 0)
        if value > 0:
            return value
    si = gpid._sync_index(rec)
    return int(si * 40000) if si >= 0 else 0


def _record_sort_key(cam: str, rec: Dict[str, Any], ordinal: int) -> Tuple[int, int, str, int]:
    si = gpid._sync_index(rec)
    ts = _record_ts_us(rec)
    if si < 0 and ts > 0:
        si = int(ts // 40000)
    return int(si if si >= 0 else 2**62), int(ts if ts > 0 else 2**62), str(cam), int(ordinal)


def _human_result_name(dataset_dir: Path, cam: str) -> str:
    for name in (f"human_result_polygon_{cam}.jsonl", f"human_result_sanitized_{cam}.jsonl"):
        if (dataset_dir / name).exists():
            return name
    return f"human_result_sanitized_{cam}.jsonl"


def _human_result_template(dataset_dir: Path, cams: List[str]) -> str:
    if any((dataset_dir / f"human_result_polygon_{cam}.jsonl").exists() for cam in cams):
        return "{root}/human_result_polygon_{camera}.jsonl"
    return "{root}/human_result_sanitized_{camera}.jsonl"


def _dataset_records(dataset_dir: Path, cams: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    heap: List[Tuple[Tuple[int, int, str, int], str, int, Iterator[Dict[str, Any]], Dict[str, Any]]] = []
    for cam in cams:
        path = dataset_dir / _human_result_name(dataset_dir, cam)
        it = _jsonl_iter(path)
        try:
            first = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (_record_sort_key(cam, first, 0), cam, 0, it, first))
    while heap:
        _key, cam, ordinal, it, rec = heapq.heappop(heap)
        yield cam, rec
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (_record_sort_key(cam, nxt, ordinal + 1), cam, ordinal + 1, it, nxt))


def _project_root_from_dataset(dataset_dir: Path) -> Path:
    parts = list(dataset_dir.resolve().parents)
    for p in [dataset_dir.resolve()] + parts:
        if (p / "global_yolo_infer_server.py").exists() and (p / "launch_profiles").exists():
            return p
    return REPO_ROOT


def _build_anchor_transform(dataset_dir: Path, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if not bool(getattr(args, "recompute_anchor", False)):
        return None
    project_root = Path(args.project_root).expanduser().resolve() if str(args.project_root or "").strip() else _project_root_from_dataset(dataset_dir)
    cams = [x.strip() for x in str(args.cams).split(",") if x.strip()]
    yargs = argparse.Namespace(
        anchor_mode=str(args.anchor_mode or "bbox_center"),
        anchor_height_correction_enable=bool(args.anchor_height_correction_enable),
        anchor_height_correction_k=float(args.anchor_height_correction_k),
    )
    projectors = {
        cam: gyolo.LocalProjector(cam, gyolo._resolve_project_path(project_root, str(args.calib_template).format(camera=cam, cam=cam, id=cam)))
        for cam in cams
    }
    nadirs = {
        cam: gyolo._load_nadir_config(cam, gyolo._resolve_project_path(project_root, str(args.anchor_nadir_pixel_template).format(camera=cam, cam=cam, id=cam)))
        for cam in cams
    }
    return {
        "project_root": str(project_root),
        "args": yargs,
        "projectors": projectors,
        "nadirs": nadirs,
        "geometry_fusion_enable": bool(args.geometry_fusion_enable),
    }


def _transform_record_anchor(cam: str, rec: Dict[str, Any], transform: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not transform:
        return rec
    out = dict(rec)
    persons = rec.get("persons")
    if not isinstance(persons, list):
        return out
    w = _safe_int(rec.get("frame_w"), 0)
    h = _safe_int(rec.get("frame_h"), 0)
    projector = transform["projectors"].get(cam)
    nadir_cfg = transform["nadirs"].get(cam, {})
    yargs = transform["args"]
    geometry = bool(transform.get("geometry_fusion_enable", False))
    new_persons: List[Dict[str, Any]] = []
    for p in persons:
        if not isinstance(p, dict):
            continue
        bbox = p.get("bbox_xyxy") or p.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            new_persons.append(dict(p))
            continue
        mask_poly = None
        mps = p.get("mask_polygons")
        if isinstance(mps, list) and mps:
            first = mps[0]
            if isinstance(first, list):
                mask_poly = first
        det = gyolo.Det(
            confidence=_safe_float(p.get("confidence"), 0.0),
            cls=0,
            bbox_xyxy=tuple(float(x) for x in bbox[:4]),
            foot_pixel=gyolo._bbox_footpoint(bbox, "bbox_center"),
            area=_safe_int(p.get("mask_area"), int(max(1.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))))),
            mask_polygon=mask_poly,
        )
        det = gyolo._apply_anchor_to_det(det, w, h, yargs, nadir_cfg)
        phys = projector.project_detail(det.foot_pixel) if projector is not None else {}
        local_xy = None
        if bool(phys.get("ok", False)):
            local_xy = [round(float(phys["x_m"]), 3), round(float(phys["y_m"]), 3)]
        nperson = dict(p)
        nperson.update({
            "anchor_pixel_raw": [round(float(det.anchor_pixel_raw[0]), 3), round(float(det.anchor_pixel_raw[1]), 3)],
            "anchor_pixel_corrected": [round(float(det.anchor_pixel_corrected[0]), 3), round(float(det.anchor_pixel_corrected[1]), 3)],
            "anchor_mode": str(det.anchor_mode),
            "anchor_quality": round(float(det.anchor_quality), 4),
            "anchor_quality_terms": dict(det.anchor_quality_terms or {}),
            "anchor_height_correction_k": round(float(det.anchor_height_correction_k), 5),
            "anchor_nadir_source": str(det.anchor_nadir_source),
            "foot_pixel": [int(round(det.foot_pixel[0])), int(round(det.foot_pixel[1]))],
            "local_xy": local_xy,
            "physical_coord": phys,
            "calibration_weight": round(float(det.calibration_weight), 4),
            "view_geometry_weight": round(float(det.view_geometry_weight), 4),
        })
        if geometry:
            nperson["fusion_weight"] = round(float(det.fusion_weight), 4)
        else:
            cw = _safe_float(nperson.get("camera_center_weight"), 1.0)
            conf = _safe_float(nperson.get("confidence"), 0.0)
            nperson["fusion_weight"] = round(max(0.01, min(1.0, conf * cw)), 4)
        new_persons.append(nperson)
    out["persons"] = new_persons
    out["person_count"] = len(new_persons)
    out["anchor_recomputed_offline"] = True
    out["anchor_recompute_geometry_fusion_enable"] = geometry
    out["anchor_recompute_mode"] = str(getattr(yargs, "anchor_mode", ""))
    return out


class ReplayClock:
    def __init__(self) -> None:
        self.now_s = real_time.time()

    def set_us(self, ts_us: int) -> None:
        if ts_us > 0:
            self.now_s = float(ts_us) / 1_000_000.0

    def time(self) -> float:
        return float(self.now_s)


def _load_profile(profile_path: Path) -> Dict[str, Any]:
    profile = _read_json(profile_path)
    if isinstance(profile.get("args"), dict):
        return profile["args"]
    return profile


def _parse_override_value(value: str) -> Any:
    text = str(value).strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        return text


def apply_profile_overrides(profile_args: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    out = dict(profile_args)
    for item in overrides:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(f"invalid override, expected key=value: {text}")
        key, raw = text.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid override key: {text}")
        if not key.startswith("global_person_id_"):
            key = "global_person_id_" + key
        out[key] = _parse_override_value(raw)
    return out


def _server_args(dataset_dir: Path, output_dir: Path, backend: str, role: str,
                 profile_args: Dict[str, Any], cams: List[str]) -> argparse.Namespace:
    human_template = _human_result_template(dataset_dir, cams)
    args = argparse.Namespace(
        cams=",".join(cams),
        sync_root=str(dataset_dir.resolve()),
        human_jsonl_template=human_template,
        output_jsonl=str((output_dir / f"{role}_{backend}_tracks.jsonl").resolve()),
        latest_json=str((output_dir / f"{role}_{backend}_latest.json").resolve()),
        status_json=str((output_dir / f"{role}_{backend}_status.json").resolve()),
        control_json=str((output_dir / f"{role}_{backend}_control.json").resolve()),
        control_poll_ms=200.0,
        stream_role=role,
        append_jsonl=False,
        start_tail=False,
        poll_ms=10.0,
        status_interval_ms=1_000_000.0,
        max_lines_per_poll=4096,
        max_sync_flush_per_loop=4096,
        sync_delay=1,
        min_confidence=0.10,
        coord_transform_mode="raw",
        field_width_m=33.0,
        field_height_m=22.0,
        field_origin_x_m=0.0,
        field_origin_y_m=0.0,
        field_margin_m=2.0,
        drop_out_of_field=False,
        max_match_speed_mps=12.0,
        max_update_speed_mps=10.0,
        publish_lost_max_age_ms=500.0,
        publish_min_hits=2,
        cluster_gate_m=0.65,
        same_cam_cluster_gate_m=0.22,
        cross_camera_cluster_gate_m=0.75,
        handoff_aware_enable=True,
        handoff_match_gate_m=1.35,
        handoff_birth_near_track_suppress_m=1.25,
        handoff_duplicate_suppress_dist_m=1.35,
        handoff_strict_count_lock_enable=True,
        handoff_strict_birth_block_m=3.0,
        handoff_strict_publish_hold_ms=1800.0,
        count_gate_enable=True,
        count_gate_x_min_m=0.0,
        count_gate_x_max_m=30.0,
        count_gate_y_min_m=0.0,
        count_gate_y_max_m=20.0,
        count_gate_short_edge_band_m=3.0,
        count_gate_short_edge_y_pad_m=2.0,
        count_gate_startup_bootstrap_ms=5000.0,
        count_gate_bootstrap_allow_anywhere_until_track_count=8,
        count_gate_non_short_edge_lost_hold_ms=5000.0,
        count_gate_non_short_edge_retire_hard_cap_ms=15000.0,
        duplicate_suppress_enable=True,
        duplicate_suppress_dist_m=0.9,
        match_gate_m=1.25,
        lost_match_gate_m=1.8,
        match_score_min=0.32,
        birth_quarantine_enable=True,
        birth_confirm_hits=2,
        birth_match_gate_m=0.85,
        birth_near_track_suppress_m=0.65,
        birth_candidate_ttl_ms=900.0,
        filter_backend=backend,
        position_alpha=0.65,
        velocity_alpha=0.25,
        kalman_use_for_association=False,
        kalman_process_noise_m=0.35,
        kalman_measurement_noise_m=0.45,
        kalman_initial_pos_var=0.6,
        kalman_initial_vel_var=4.0,
        kalman_max_innovation_m=2.5,
        kalman_reject_innovation_enable=False,
        confirm_hits=2,
        miss_grace_ms=180.0,
        lost_after_ms=650.0,
        reid_memory_ms=3000.0,
        retire_after_ms=12000.0,
        local_id_bonus=0.10,
        local_id_bonus_max=0.18,
        id_switch_suspect_gate_m=0.80,
        history_len=64,
        output_history_len=16,
    )
    args.sync_root = str(dataset_dir.resolve())
    args.human_jsonl_template = human_template
    args.output_jsonl = str((output_dir / f"{role}_{backend}_tracks.jsonl").resolve())
    args.latest_json = str((output_dir / f"{role}_{backend}_latest.json").resolve())
    args.status_json = str((output_dir / f"{role}_{backend}_status.json").resolve())
    args.control_json = str((output_dir / f"{role}_{backend}_control.json").resolve())
    args.append_jsonl = False
    args.start_tail = False
    args.stream_role = role
    args.filter_backend = backend
    args.status_interval_ms = 1_000_000.0
    args.max_lines_per_poll = 4096
    args.max_sync_flush_per_loop = 4096
    for dest in vars(args).keys():
        key = f"global_person_id_{dest}"
        if key in profile_args:
            setattr(args, dest, profile_args[key])
    args.output_jsonl = str((output_dir / f"{role}_{backend}_tracks.jsonl").resolve())
    args.latest_json = str((output_dir / f"{role}_{backend}_latest.json").resolve())
    args.status_json = str((output_dir / f"{role}_{backend}_status.json").resolve())
    args.control_json = str((output_dir / f"{role}_{backend}_control.json").resolve())
    args.human_jsonl_template = human_template
    args.sync_root = str(dataset_dir.resolve())
    args.start_tail = False
    args.append_jsonl = False
    args.stream_role = role
    args.filter_backend = backend
    return args


def _feed_record(server: gpid.GlobalPersonIdServer, cam: str, rec: Dict[str, Any]) -> None:
    server.stats["records"] += 1
    si = gpid._sync_index(rec)
    if si < 0:
        ts_us = _record_ts_us(rec)
        si = int(ts_us // 40000) if ts_us > 0 else -1
    if si >= 0:
        server.latest_input_sync_index = max(int(server.latest_input_sync_index), int(si))
    dets = server._record_to_detections(cam, rec)
    if not dets:
        if si >= 0 and server.tracks:
            server.pending.setdefault(int(si), [])
        return
    server.stats["detections"] += len(dets)
    si = dets[0].sync_index
    if si < 0:
        si = int(dets[0].timestamp_us // 40000)
    server.pending.setdefault(int(si), []).extend(dets)


def _drain_pending(server: gpid.GlobalPersonIdServer) -> None:
    while server.pending:
        si = sorted(server.pending.keys())[0]
        dets = server.pending.pop(si, [])
        obs = server._cluster_detections(si, dets)
        server.stats["observations"] += len(obs)
        server._assign(obs)
        server._publish(si)


def run_backend(dataset_dir: Path, output_dir: Path, backend: str, role: str,
                profile_args: Dict[str, Any], cams: List[str],
                anchor_transform: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _server_args(dataset_dir, output_dir, backend, role, profile_args, cams)
    clock = ReplayClock()
    saved_time = gpid.time.time
    gpid.time.time = clock.time
    started = real_time.time()
    server: Optional[gpid.GlobalPersonIdServer] = None
    try:
        server = gpid.GlobalPersonIdServer(args)
        server._publish(-1)
        for cam, rec in _dataset_records(dataset_dir, cams):
            rec = _transform_record_anchor(cam, rec, anchor_transform)
            ts_us = _record_ts_us(rec)
            clock.set_us(ts_us)
            _feed_record(server, cam, rec)
            server._flush_ready_syncs()
        _drain_pending(server)
        server._write_status()
        return {
            "backend": backend,
            "role": role,
            "output_jsonl": str(Path(args.output_jsonl)),
            "latest_json": str(Path(args.latest_json)),
            "status_json": str(Path(args.status_json)),
            "runtime_s": round(real_time.time() - started, 3),
            "stats": dict(server.stats),
        }
    finally:
        if server is not None:
            try:
                server.out_fp.close()
            except Exception:
                pass
        gpid.time.time = saved_time


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * float(p)))))
    return float(vals[idx])


def summarize_tracks(path: Path, expected_count: int) -> Dict[str, Any]:
    rows = 0
    counts: List[int] = []
    unique_ids = set()
    duplicate_close_rows = 0
    per_id: Dict[int, Dict[str, Any]] = {}
    last_stats: Dict[str, Any] = {}
    prev_by_id: Dict[int, Tuple[int, float, float]] = {}
    step_values: List[float] = []
    speed_values: List[float] = []
    for obj in _jsonl_iter(path):
        rows += 1
        tracks = obj.get("tracks") if isinstance(obj.get("tracks"), list) else []
        counts.append(len(tracks))
        if isinstance(obj.get("stats"), dict):
            last_stats = obj["stats"]
        points: List[Tuple[int, float, float]] = []
        ts = _safe_int(obj.get("timestamp_us"), 0)
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            gid = _safe_int(tr.get("global_id"), -1)
            if gid < 0:
                text = str(tr.get("global_player_id", "")).lstrip("P")
                gid = _safe_int(text, -1)
            if gid < 0:
                continue
            xy = tr.get("field_xy") or tr.get("filtered_xy") or [tr.get("x_m"), tr.get("y_m")]
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            x, y = _safe_float(xy[0]), _safe_float(xy[1])
            unique_ids.add(gid)
            item = per_id.setdefault(gid, {
                "rows": 0,
                "first_ts_us": ts,
                "last_ts_us": ts,
                "distance_m": 0.0,
                "max_gap_ms": 0.0,
                "states": {},
            })
            item["rows"] += 1
            item["last_ts_us"] = ts
            item["states"][str(tr.get("state", ""))] = item["states"].get(str(tr.get("state", "")), 0) + 1
            prev = prev_by_id.get(gid)
            if prev is not None:
                pts, px, py = prev
                dt_ms = max(0.0, (ts - pts) / 1000.0)
                dist = math.hypot(x - px, y - py)
                item["distance_m"] += dist
                item["max_gap_ms"] = max(float(item["max_gap_ms"]), dt_ms)
                step_values.append(dist)
                if dt_ms > 0:
                    speed_values.append(dist / (dt_ms / 1000.0))
            prev_by_id[gid] = (ts, x, y)
            points.append((gid, x, y))
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if math.hypot(points[i][1] - points[j][1], points[i][2] - points[j][2]) <= 0.9:
                    duplicate_close_rows += 1
                    break
            else:
                continue
            break
    exact = sum(1 for c in counts if c == expected_count)
    under = sum(1 for c in counts if c < expected_count)
    over = sum(1 for c in counts if c > expected_count)
    count_n = max(1, len(counts))
    return {
        "path": str(path),
        "rows": rows,
        "expected_count": expected_count,
        "track_count_mean": round(sum(counts) / count_n, 3) if counts else 0.0,
        "track_count_max": max(counts) if counts else 0,
        "track_count_p95": round(_percentile([float(c) for c in counts], 0.95), 3),
        "expected_exact_pct": round(exact * 100.0 / count_n, 2) if counts else 0.0,
        "under_expected_pct": round(under * 100.0 / count_n, 2) if counts else 0.0,
        "over_expected_pct": round(over * 100.0 / count_n, 2) if counts else 0.0,
        "unique_published_ids": len(unique_ids),
        "fragmentation_ids_over_expected": max(0, len(unique_ids) - max(0, expected_count)),
        "duplicate_close_rows": duplicate_close_rows,
        "step_m_mean": round(sum(step_values) / max(1, len(step_values)), 4) if step_values else 0.0,
        "step_m_p95": round(_percentile(step_values, 0.95), 4),
        "speed_mps_p95": round(_percentile(speed_values, 0.95), 4),
        "last_stats": last_stats,
        "per_id": per_id,
    }


def infer_expected_count(dataset_dir: Path, manifest: Dict[str, Any]) -> int:
    text = " ".join([
        str(dataset_dir.name).lower(),
        str(manifest.get("label", "")).lower(),
        str(manifest.get("description", "")).lower(),
        str(manifest.get("dataset_type", "")).lower(),
    ])
    if "negative" in text or "no_person" in text or "empty" in text:
        return 0
    if "single" in text or "one_person" in text:
        return 1
    if "two" in text or "2person" in text or "two_person" in text:
        return 2
    return 1


def compare_dataset(dataset_dir: Path, output_root: Path, profile_args: Dict[str, Any],
                    cams: List[str], expected_count: Optional[int],
                    cli_args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    manifest = _read_json(dataset_dir / "manifest.json")
    summary = _read_json(dataset_dir / "summary.json")
    expected = int(expected_count if expected_count is not None else infer_expected_count(dataset_dir, manifest))
    out_dir = output_root / dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_transform = _build_anchor_transform(dataset_dir, cli_args) if cli_args is not None else None
    runs = [
        run_backend(dataset_dir, out_dir, "alpha_beta", "official_replay", profile_args, cams, anchor_transform),
        run_backend(dataset_dir, out_dir, "kalman_cv", "shadow_replay", profile_args, cams, anchor_transform),
    ]
    metrics: Dict[str, Any] = {}
    baseline = dataset_dir / "global_person_tracks_baseline.jsonl"
    if baseline.exists() and baseline.stat().st_size > 0:
        metrics["recorded_baseline"] = summarize_tracks(baseline, expected)
    for run in runs:
        metrics[f"{run['role']}_{run['backend']}"] = summarize_tracks(Path(run["output_jsonl"]), expected)
    result = {
        "schema_version": 1,
        "kind": "global_person_id_filter_compare",
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "manifest": manifest,
        "record_summary": summary,
        "expected_count": expected,
        "anchor_recompute": {
            "enabled": bool(anchor_transform),
            "mode": str(getattr(cli_args, "anchor_mode", "")) if cli_args is not None else "",
            "height_correction_enable": bool(getattr(cli_args, "anchor_height_correction_enable", False)) if cli_args is not None else False,
            "height_correction_k": float(getattr(cli_args, "anchor_height_correction_k", 0.0)) if cli_args is not None else 0.0,
            "geometry_fusion_enable": bool(getattr(cli_args, "geometry_fusion_enable", False)) if cli_args is not None else False,
        },
        "runs": runs,
        "metrics": metrics,
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "comparison_report.md").write_text(render_dataset_report(result), encoding="utf-8")
    return result


def _metric_row(name: str, m: Dict[str, Any]) -> str:
    stats = m.get("last_stats", {}) if isinstance(m.get("last_stats"), dict) else {}
    return (
        f"| {name} | {m.get('rows', 0)} | {m.get('track_count_mean', 0)} | "
        f"{m.get('track_count_p95', 0)} | {m.get('expected_exact_pct', 0)} | "
        f"{m.get('under_expected_pct', 0)} | {m.get('over_expected_pct', 0)} | "
        f"{m.get('unique_published_ids', 0)} | {m.get('fragmentation_ids_over_expected', 0)} | "
        f"{m.get('duplicate_close_rows', 0)} | {stats.get('created_tracks', '-')} | "
        f"{stats.get('birth_candidates_promoted', '-')} | {stats.get('birth_suppressed_near_track', '-')} | "
        f"{stats.get('id_switch_suspects', '-')} | {m.get('step_m_p95', 0)} | {m.get('speed_mps_p95', 0)} |"
    )


def render_dataset_report(result: Dict[str, Any]) -> str:
    lines = [
        f"# Global Person ID Filter Compare - {result.get('dataset_name')}",
        "",
        f"- dataset_dir: `{result.get('dataset_dir')}`",
        f"- expected_count: `{result.get('expected_count')}`",
        f"- label: `{(result.get('manifest') or {}).get('label', '')}`",
        f"- anchor_recompute: `{(result.get('anchor_recompute') or {}).get('enabled', False)}`",
        f"- anchor_mode: `{(result.get('anchor_recompute') or {}).get('mode', '')}`",
        f"- geometry_fusion_enable: `{(result.get('anchor_recompute') or {}).get('geometry_fusion_enable', False)}`",
        "",
        "| run | rows | mean_count | p95_count | exact_% | under_% | over_% | unique_ids | frag_over | dup_close_rows | created | birth_promoted | birth_suppressed | id_switch_suspects | step_p95_m | speed_p95_mps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (result.get("metrics") or {}).items():
        if isinstance(metrics, dict):
            lines.append(_metric_row(name, metrics))
    lines.append("")
    lines.append("Notes:")
    lines.append("- `exact_%` is the share of published rows whose track_count equals the expected person count inferred from the dataset name/manifest.")
    lines.append("- `frag_over` is unique published IDs beyond expected count; lower is better for ID continuity.")
    lines.append("- `step_p95_m` and `speed_p95_mps` are coarse smoothness indicators, not physical ground truth.")
    lines.append("")
    return "\n".join(lines)


def render_index_report(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Global Person ID Official vs Shadow Dataset Compare",
        "",
        "| dataset | expected | run | exact_% | under_% | over_% | unique_ids | frag_over | dup_close_rows | created | birth_promoted | birth_suppressed | id_switch_suspects | step_p95_m | speed_p95_mps |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        dataset = result.get("dataset_name", "")
        expected = result.get("expected_count", "")
        for name, m in (result.get("metrics") or {}).items():
            if not isinstance(m, dict):
                continue
            stats = m.get("last_stats", {}) if isinstance(m.get("last_stats"), dict) else {}
            lines.append(
                f"| {dataset} | {expected} | {name} | {m.get('expected_exact_pct', 0)} | "
                f"{m.get('under_expected_pct', 0)} | {m.get('over_expected_pct', 0)} | "
                f"{m.get('unique_published_ids', 0)} | {m.get('fragmentation_ids_over_expected', 0)} | "
                f"{m.get('duplicate_close_rows', 0)} | {stats.get('created_tracks', '-')} | "
                f"{stats.get('birth_candidates_promoted', '-')} | {stats.get('birth_suppressed_near_track', '-')} | "
                f"{stats.get('id_switch_suspects', '-')} | {m.get('step_m_p95', 0)} | {m.get('speed_mps_p95', 0)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Global Person ID official/shadow filters on recorded datasets.")
    ap.add_argument("--dataset-dir", action="append", default=[], help="Dataset folder containing human_result_sanitized_<cam>.jsonl or human_result_polygon_<cam>.jsonl.")
    ap.add_argument("--dataset-root", default="", help="Optional root; when set, compare latest matching folders.")
    ap.add_argument("--name-contains", action="append", default=[], help="Filter dataset-root folders by substring.")
    ap.add_argument("--profile", default=str(REPO_ROOT / "launch_profiles" / "627_event_overlay_bev.json"))
    ap.add_argument("--override", action="append", default=[], help="Override profile args, e.g. birth_near_track_suppress_m=0.55")
    ap.add_argument("--output-root", default="")
    ap.add_argument("--cams", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument("--project-root", default="")
    ap.add_argument("--recompute-anchor", action="store_true", default=False)
    ap.add_argument("--anchor-mode", choices=["bbox_center", "bbox_bottom", "mask_robust_center", "mask_eroded_centroid", "auto_topdown_center"], default="bbox_center")
    ap.add_argument("--anchor-height-correction-enable", action="store_true", default=True)
    ap.add_argument("--no-anchor-height-correction-enable", dest="anchor_height_correction_enable", action="store_false")
    ap.add_argument("--anchor-height-correction-k", type=float, default=0.08)
    ap.add_argument("--anchor-nadir-pixel-template", default="calib_true/nadir_camera{camera}.json")
    ap.add_argument("--calib-template", default="calib_true/field_calib_camera{camera}.json")
    ap.add_argument("--geometry-fusion-enable", action="store_true", default=False)
    ap.add_argument("--no-geometry-fusion-enable", dest="geometry_fusion_enable", action="store_false")
    args = ap.parse_args()

    cams = [x.strip() for x in str(args.cams).split(",") if x.strip()]
    profile_args = apply_profile_overrides(_load_profile(Path(args.profile)), args.override)
    datasets = [Path(p).expanduser().resolve() for p in args.dataset_dir]
    if args.dataset_root:
        root = Path(args.dataset_root).expanduser().resolve()
        filters = [str(x).lower() for x in args.name_contains if str(x).strip()]
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if filters and not any(f in child.name.lower() for f in filters):
                continue
            if (child / "manifest.json").exists():
                datasets.append(child)
    if not datasets:
        raise SystemExit("no dataset dirs provided")

    stamp = real_time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser() if args.output_root else (datasets[0].parent / f"_filter_compare_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    results = [compare_dataset(ds, output_root, profile_args, cams, args.expected_count, args) for ds in datasets]
    index = {
        "schema_version": 1,
        "kind": "global_person_id_filter_compare_index",
        "created_wall_us": int(real_time.time() * 1_000_000),
        "profile": str(Path(args.profile).expanduser().resolve()),
        "output_root": str(output_root.resolve()),
        "datasets": results,
    }
    (output_root / "comparison_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "comparison_report.md").write_text(render_index_report(results), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "output_root": str(output_root.resolve()),
        "dataset_count": len(results),
        "report": str((output_root / "comparison_report.md").resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
