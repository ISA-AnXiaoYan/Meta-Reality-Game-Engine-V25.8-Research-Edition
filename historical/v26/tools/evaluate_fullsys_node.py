#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _dig(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _compact_from_full(report: Dict[str, Any]) -> Dict[str, Any]:
    runtime = report.get("latest_runtime_snapshot")
    if not isinstance(runtime, dict) or not runtime:
        runtime = report.get("latest_snapshot", {})
    if not isinstance(runtime, dict):
        runtime = {}
    gb = runtime.get("global_bev", {}) if isinstance(runtime.get("global_bev"), dict) else {}
    gy = runtime.get("global_yolo", {}) if isinstance(runtime.get("global_yolo"), dict) else {}
    layer = gb.get("event_layer", {}) if isinstance(gb.get("event_layer"), dict) else {}
    layer_presentation = layer.get("presentation_worker", {}) if isinstance(layer.get("presentation_worker"), dict) else {}
    alignment = gb.get("alignment", {}) if isinstance(gb.get("alignment"), dict) else {}
    quality = alignment.get("quality", {}) if isinstance(alignment.get("quality"), dict) else {}
    audit = report.get("status_key_audit", {}) if isinstance(report.get("status_key_audit"), dict) else {}
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
        "status_freshness": report.get("status_freshness", {}),
        "status_key_ok": audit.get("ok"),
        "mask_stats_jsonl_no_growth": report.get("mask_stats_jsonl_no_growth"),
        "final_snapshot_is_after_shutdown": report.get("final_snapshot_is_after_shutdown"),
        "global_bev": {
            "render_fps": gb.get("render_fps"),
            "output_fps": gb.get("output_fps"),
            "event_layer_dilate_backend": layer.get("dilate_backend"),
            "event_layer_dilate_px": layer.get("dilate_px"),
            "mvs_texture_upload": gb.get("mvs_texture_upload"),
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
                "overall_state": alignment.get("overall_state"),
                "mvs_synced_cam_count": quality.get("mvs_synced_cam_count"),
                "mvs_missing_history_cam_count": quality.get("mvs_missing_history_cam_count"),
                "mvs_metadata_outlier_cam_count": quality.get("mvs_metadata_outlier_cam_count"),
                "event_synced_cam_count": quality.get("event_synced_cam_count"),
                "event_sync_outlier_cam_count": quality.get("event_sync_outlier_cam_count"),
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
            },
        },
        "global_yolo": {
            "worker_count": gy.get("worker_count"),
            "fps": gy.get("fps"),
            "inferred_fps_total": gy.get("inferred_fps_total"),
            "published_fps_total": gy.get("published_fps_total"),
            "async_record_postprocess_enabled": gy.get("async_record_postprocess_enabled"),
            "postprocess_queue_depth": gy.get("postprocess_queue_depth"),
            "postprocess_lag_ms": gy.get("postprocess_lag_ms"),
            "postprocess_backpressure_ms": gy.get("postprocess_backpressure_ms"),
            "postprocess_dropped_batches": gy.get("postprocess_dropped_batches"),
            "polygon_contract_ok": gy.get("polygon_contract_ok"),
            "published_records_with_polygon_ratio": gy.get("published_records_with_polygon_ratio"),
            "published_persons_with_polygon_ratio": gy.get("published_persons_with_polygon_ratio"),
            "input_alignment_audit": gy.get("input_alignment_audit"),
            "input_buffer": gy.get("input_buffer"),
            "summary": gy.get("summary"),
        },
    }


def _normalize(report: Dict[str, Any]) -> Dict[str, Any]:
    if report.get("kind") == "fullsys_test_report":
        return _compact_from_full(report)
    return report


def _add(items: List[Dict[str, str]], level: str, key: str, message: str) -> None:
    items.append({"level": level, "key": key, "message": message})


def evaluate(report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    compact = _normalize(report)
    findings: List[Dict[str, str]] = []

    if compact.get("launcher_clean_exit") is not True:
        _add(findings, "FAIL", "launcher_clean_exit", "launcher did not exit cleanly")
    if compact.get("launcher_timeout_ok") is not True and str(compact.get("launcher_rc")) != "0":
        _add(findings, "FAIL", "launcher_rc", f"unexpected launcher rc={compact.get('launcher_rc')}")
    if compact.get("runtime_overall_state") != "OK":
        _add(findings, "FAIL", "runtime_overall_state", f"runtime state is {compact.get('runtime_overall_state')}")
    if compact.get("runtime_broken_links"):
        _add(findings, "FAIL", "runtime_broken_links", f"broken links: {compact.get('runtime_broken_links')}")
    if compact.get("status_key_ok") is not True:
        _add(findings, "FAIL", "status_key_ok", "status key audit failed or is missing")
    if _dig(compact, "status_freshness.global_bev_from_this_run") is not True:
        _add(findings, "FAIL", "status_freshness.global_bev_from_this_run", "Global BEV status was not proven to be from this run")

    if compact.get("runtime_degraded_links"):
        _add(findings, "WARN", "runtime_degraded_links", f"degraded links: {compact.get('runtime_degraded_links')}")
    if compact.get("mask_stats_jsonl_no_growth") is not True:
        _add(findings, "WARN", "mask_stats_jsonl_no_growth", "event_human_mask_stats jsonl changed during run")

    render_fps = _as_float(_dig(compact, "global_bev.render_fps"))
    output_fps = _as_float(_dig(compact, "global_bev.output_fps"))
    if render_fps and render_fps < args.bev_render_fail_fps:
        _add(findings, "FAIL", "global_bev.render_fps", f"render_fps={render_fps:.3f} below fail threshold {args.bev_render_fail_fps}")
    elif render_fps and render_fps < args.bev_render_warn_fps:
        _add(findings, "WARN", "global_bev.render_fps", f"render_fps={render_fps:.3f} below warning threshold {args.bev_render_warn_fps}")
    if output_fps and output_fps < args.bev_output_warn_fps:
        _add(findings, "WARN", "global_bev.output_fps", f"output_fps={output_fps:.3f} below warning threshold {args.bev_output_warn_fps}")

    mvs_synced = _as_int(_dig(compact, "global_bev.alignment.mvs_synced_cam_count"))
    if mvs_synced and mvs_synced < args.required_mvs_synced_cams:
        _add(findings, "FAIL", "alignment.mvs_synced_cam_count", f"mvs_synced_cam_count={mvs_synced}")
    if _as_int(_dig(compact, "global_bev.alignment.mvs_missing_history_cam_count")) > 0:
        _add(findings, "FAIL", "alignment.mvs_missing_history_cam_count", "MVS missing-history cameras present")
    if _as_int(_dig(compact, "global_bev.alignment.mvs_metadata_outlier_cam_count")) > 0:
        _add(findings, "FAIL", "alignment.mvs_metadata_outlier_cam_count", "MVS metadata-outlier cameras present")

    event_outliers = _as_int(_dig(compact, "global_bev.alignment.event_sync_outlier_cam_count"))
    if event_outliers > 0:
        _add(findings, "WARN", "alignment.event_sync_outlier_cam_count", f"Event-to-BEV sync outlier cameras={event_outliers}")

    if args.require_v24c6_keys:
        if not isinstance(_dig(compact, "global_bev.mvs_texture_upload"), dict):
            _add(findings, "FAIL", "global_bev.mvs_texture_upload", "missing V24-C5 mvs_texture_upload object")
        if _dig(compact, "global_bev.event_layer_display_budget_ms") is None:
            _add(findings, "FAIL", "global_bev.event_layer_display_budget_ms", "missing V24-C6 event display budget field")

    mvs_upload_elapsed = _as_float(_dig(compact, "global_bev.mvs_texture_upload.elapsed_ms"))
    mvs_upload_budget = _as_float(_dig(compact, "global_bev.mvs_texture_upload.budget_ms"))
    mvs_held = _as_int(_dig(compact, "global_bev.mvs_texture_upload.held_by_budget_cam_count"))
    if mvs_held > 0:
        _add(findings, "WARN", "mvs_texture_upload.held_by_budget_cam_count", f"MVS texture held cameras={mvs_held}")
    if mvs_upload_budget > 0 and mvs_upload_elapsed > mvs_upload_budget * args.budget_overrun_warn_ratio:
        _add(findings, "WARN", "mvs_texture_upload.elapsed_ms", f"MVS upload elapsed={mvs_upload_elapsed:.3f}ms over budget={mvs_upload_budget:.3f}ms")

    event_elapsed = _as_float(_dig(compact, "global_bev.event_layer_display_budget_elapsed_ms"))
    event_budget = _as_float(_dig(compact, "global_bev.event_layer_display_budget_ms"))
    event_held = _as_int(_dig(compact, "global_bev.event_layer_display_budget_held_cam_count"))
    event_overrun_rate = _as_float(_dig(compact, "global_bev.event_layer_display_budget_overrun_rate"))
    pres_enabled = _dig(compact, "global_bev.event_layer_presentation_worker.enabled")
    pres_mode = str(_dig(compact, "global_bev.event_layer_presentation_worker.mode", "") or "")
    pres_state = str(_dig(compact, "global_bev.event_layer_presentation_worker.state", "") or "")
    pres_worker_state = str(_dig(compact, "global_bev.event_layer_presentation_worker.worker_state", "") or "")
    pres_compare = str(_dig(compact, "global_bev.event_layer_presentation_worker.compare_state", "") or "")
    pres_mismatch = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.mismatch_cam_count"))
    pres_payload_policy = str(_dig(compact, "global_bev.event_layer_presentation_worker.payload_policy", "") or "")
    pres_read_points = _dig(compact, "global_bev.event_layer_presentation_worker.read_points")
    pres_display_compare = str(_dig(compact, "global_bev.event_layer_presentation_worker.display_compare_state", pres_compare) or "")
    pres_display_mismatch = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.display_mismatch_cam_count", pres_mismatch))
    pres_delta_compare = str(_dig(compact, "global_bev.event_layer_presentation_worker.delta_compare_state", "") or "")
    pres_delta_mismatch = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.delta_mismatch_cam_count"))
    pres_compare_lookup_state = str(_dig(compact, "global_bev.event_layer_presentation_worker.compare_lookup_state", "") or "")
    pres_compare_target_delta_ticks = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.compare_target_delta_ticks"), -999999)
    pres_compare_tolerance_ticks = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.compare_tolerance_ticks"), -1)
    pres_compare_exact_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.compare_exact_count"))
    pres_compare_near_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.compare_near_count"))
    pres_compare_miss_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.compare_miss_count"))
    pres_compare_hit_ratio = _as_float(_dig(compact, "global_bev.event_layer_presentation_worker.compare_hit_ratio"))
    pres_build_p95 = _as_float(_dig(compact, "global_bev.event_layer_presentation_worker.build_p95_ms"))
    pres_fallback = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.fallback_count"))
    pres_active_consume_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.active_consume_count"))
    pres_bundle_hit_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.bundle_hit_count"))
    pres_bundle_miss_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.bundle_miss_count"))
    pres_bundle_hit_ratio = _as_float(_dig(compact, "global_bev.event_layer_presentation_worker.bundle_hit_ratio"))
    pres_consume_exact_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.consume_exact_count"))
    pres_consume_near_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.consume_near_count"))
    pres_bundle_ring_count = _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.bundle_ring_count"))
    if event_held > 0:
        _add(findings, "WARN", "event_layer.display_budget_held_cam_count", f"Event layer display-held cameras={event_held}")
    if event_budget > 0 and event_elapsed > event_budget * args.budget_overrun_warn_ratio:
        _add(findings, "WARN", "event_layer.display_budget_elapsed_ms", f"Event display elapsed={event_elapsed:.3f}ms over budget={event_budget:.3f}ms")
    node_upper = str(args.node or "").upper()
    is_v24e4 = "V24-E4" in node_upper or "V24E4" in node_upper
    if is_v24e4:
        if pres_enabled is not True:
            _add(findings, "FAIL", "event_layer.presentation_worker.enabled", f"presentation worker enabled={pres_enabled}")
        if "E4A" in node_upper and pres_mode != "shadow":
            _add(findings, "FAIL", "event_layer.presentation_worker.mode", f"expected shadow for V24-E4A, got {pres_mode}")
        if "E4B" in node_upper and pres_mode != "active":
            _add(findings, "FAIL", "event_layer.presentation_worker.mode", f"expected active for V24-E4B, got {pres_mode}")
        if "E4A" in node_upper:
            if pres_payload_policy and pres_payload_policy != "metadata_only_no_vertices":
                _add(findings, "WARN", "event_layer.presentation_worker.payload_policy", f"shadow payload_policy={pres_payload_policy}")
            if pres_read_points is True:
                _add(findings, "WARN", "event_layer.presentation_worker.read_points", "shadow should not read sparse point payloads")
            if pres_display_compare == "mismatch":
                _add(findings, "FAIL", "event_layer.presentation_worker.display_compare_state", f"shadow display_compare_state={pres_display_compare}")
            elif pres_display_compare != "match":
                _add(findings, "WARN", "event_layer.presentation_worker.display_compare_state", f"shadow display_compare_state={pres_display_compare}, lookup={pres_compare_lookup_state}, hit_ratio={pres_compare_hit_ratio:.4f}")
            if pres_display_mismatch > 0:
                _add(findings, "FAIL", "event_layer.presentation_worker.display_mismatch_cam_count", f"shadow display mismatch cameras={pres_display_mismatch}")
            if pres_delta_mismatch > 0:
                _add(findings, "WARN", "event_layer.presentation_worker.delta_mismatch_cam_count", f"shadow delta diagnostic mismatch cameras={pres_delta_mismatch}")
        if "E4B" in node_upper:
            if pres_state != "active_consumed" and pres_active_consume_count <= 0:
                _add(findings, "WARN", "event_layer.presentation_worker.state", f"active state={pres_state}, consume_count={pres_active_consume_count}")
            if pres_display_mismatch > 0:
                _add(findings, "WARN", "event_layer.presentation_worker.display_mismatch_cam_count", f"active display mismatch cameras={pres_display_mismatch}")
            if pres_fallback > 0:
                _add(findings, "WARN", "event_layer.presentation_worker.fallback_count", f"active fallback count={pres_fallback}")
        if event_budget > 0 and pres_build_p95 > event_budget * args.budget_overrun_warn_ratio:
            _add(findings, "WARN", "event_layer.presentation_worker.build_p95_ms", f"presentation build p95={pres_build_p95:.3f}ms over event budget={event_budget:.3f}ms")

    is_v24e5 = "V24-E5" in node_upper or "V24E5" in node_upper
    if is_v24e5:
        if pres_enabled is not True:
            _add(findings, "FAIL", "event_layer.presentation_worker.enabled", f"presentation worker enabled={pres_enabled}")
        if "E5A" in node_upper and pres_mode != "shadow":
            _add(findings, "FAIL", "event_layer.presentation_worker.mode", f"expected shadow for V24-E5A, got {pres_mode}")
        if "E5B" in node_upper and pres_mode != "active":
            _add(findings, "FAIL", "event_layer.presentation_worker.mode", f"expected active for V24-E5B, got {pres_mode}")
        if pres_bundle_ring_count <= 0:
            _add(findings, "FAIL", "event_layer.presentation_worker.bundle_ring_count", f"bundle_ring_count={pres_bundle_ring_count}")
        if "E5A" in node_upper:
            if pres_payload_policy and pres_payload_policy != "metadata_only_no_vertices":
                _add(findings, "WARN", "event_layer.presentation_worker.payload_policy", f"shadow payload_policy={pres_payload_policy}")
            if pres_display_compare == "mismatch":
                _add(findings, "FAIL", "event_layer.presentation_worker.display_compare_state", f"shadow display_compare_state={pres_display_compare}")
            elif pres_display_compare != "match":
                _add(findings, "WARN", "event_layer.presentation_worker.display_compare_state", f"shadow display_compare_state={pres_display_compare}, lookup={pres_compare_lookup_state}, hit_ratio={pres_compare_hit_ratio:.4f}")
            if pres_display_mismatch > 0:
                _add(findings, "FAIL", "event_layer.presentation_worker.display_mismatch_cam_count", f"shadow display mismatch cameras={pres_display_mismatch}")
        if "E5B" in node_upper:
            if pres_active_consume_count <= 0:
                _add(findings, "WARN", "event_layer.presentation_worker.active_consume_count", f"active consume count={pres_active_consume_count}")
            total_consume = int(pres_active_consume_count) + int(pres_fallback)
            fallback_ratio = float(pres_fallback) / float(total_consume) if total_consume > 0 else 0.0
            if pres_bundle_hit_ratio > 0.0 and pres_bundle_hit_ratio < 0.80:
                _add(findings, "WARN", "event_layer.presentation_worker.bundle_hit_ratio", f"bundle hit ratio={pres_bundle_hit_ratio:.4f}")
            if total_consume > 0 and fallback_ratio > 0.20:
                _add(findings, "WARN", "event_layer.presentation_worker.fallback_ratio", f"fallback ratio={fallback_ratio:.4f} ({pres_fallback}/{total_consume})")
            if pres_display_mismatch > 0:
                _add(findings, "WARN", "event_layer.presentation_worker.display_mismatch_cam_count", f"active display mismatch cameras={pres_display_mismatch}")

    is_v24z = "V24-Z" in node_upper or "V24Z" in node_upper
    yolo_published_fps = _as_float(_dig(compact, "global_yolo.published_fps_total"), _as_float(_dig(compact, "global_yolo.fps")))
    yolo_inferred_fps = _as_float(_dig(compact, "global_yolo.inferred_fps_total"))
    yolo_queue_depth = _as_int(_dig(compact, "global_yolo.postprocess_queue_depth"))
    yolo_dropped = _as_int(_dig(compact, "global_yolo.postprocess_dropped_batches"))
    yolo_polygon_ok = _dig(compact, "global_yolo.polygon_contract_ok")
    yolo_async = _dig(compact, "global_yolo.async_record_postprocess_enabled")
    yolo_person_polygon_ratio = _as_float(_dig(compact, "global_yolo.published_persons_with_polygon_ratio"), 1.0)
    yolo_input_align_enabled = _dig(compact, "global_yolo.input_alignment_audit.enabled")
    yolo_input_align_full_ratio = _as_float(_dig(compact, "global_yolo.input_alignment_audit.full_batch_ratio"))
    yolo_input_align_same_sync_ratio = _as_float(_dig(compact, "global_yolo.input_alignment_audit.same_sync_batch_ratio"))
    yolo_input_align_current_full_ratio = _as_float(_dig(compact, "global_yolo.input_alignment_audit.current_aligned_full_batch_ratio"))
    yolo_input_align_within_wait_ratio = _as_float(_dig(compact, "global_yolo.input_alignment_audit.sync_full_sets_within_wait_ratio"))
    yolo_input_align_wait_avg_ms = _as_float(_dig(compact, "global_yolo.input_alignment_audit.sync_full_wait_ms_avg"))
    yolo_input_align_visual_spread_max_ms = _as_float(_dig(compact, "global_yolo.input_alignment_audit.visual_mid_spread_ms_max"))
    yolo_input_align_visual_spread_p95_ms = _as_float(_dig(compact, "global_yolo.input_alignment_audit.visual_mid_spread_ms_p95"))
    yolo_input_align_exposure_spread_p95_us = _as_int(_dig(compact, "global_yolo.input_alignment_audit.exposure_spread_us_p95"))
    yolo_input_buffer_enabled = _dig(compact, "global_yolo.input_buffer.enabled")
    yolo_input_buffer_aligned_full_ratio = _as_float(_dig(compact, "global_yolo.input_buffer.aligned_full_ratio"))
    yolo_input_buffer_fallback_ratio = _as_float(_dig(compact, "global_yolo.input_buffer.fallback_ratio"))
    if is_v24z:
        if yolo_async is not True:
            _add(findings, "FAIL", "global_yolo.async_record_postprocess_enabled", "V24-Z async record postprocess is not enabled")
        if yolo_dropped > 0:
            _add(findings, "FAIL", "global_yolo.postprocess_dropped_batches", f"postprocess dropped batches={yolo_dropped}")
        if yolo_polygon_ok is not True:
            _add(findings, "FAIL", "global_yolo.polygon_contract_ok", f"polygon contract status is {yolo_polygon_ok}")
        if yolo_published_fps and yolo_published_fps < args.yolo_published_fail_fps:
            _add(findings, "FAIL", "global_yolo.published_fps_total", f"published_fps_total={yolo_published_fps:.3f} below fail threshold {args.yolo_published_fail_fps}")
        elif yolo_published_fps and yolo_published_fps < args.yolo_published_warn_fps:
            _add(findings, "WARN", "global_yolo.published_fps_total", f"published_fps_total={yolo_published_fps:.3f} below warning threshold {args.yolo_published_warn_fps}")
        if yolo_queue_depth > args.yolo_queue_depth_warn:
            _add(findings, "WARN", "global_yolo.postprocess_queue_depth", f"postprocess_queue_depth={yolo_queue_depth}")
        if yolo_person_polygon_ratio < args.yolo_polygon_ratio_warn:
            _add(findings, "WARN", "global_yolo.published_persons_with_polygon_ratio", f"person polygon ratio={yolo_person_polygon_ratio:.4f}")

    levels = {item["level"] for item in findings}
    overall = "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "PASS")
    metrics = {
        "render_fps": render_fps,
        "output_fps": output_fps,
        "mvs_synced_cam_count": mvs_synced,
        "event_sync_outlier_cam_count": event_outliers,
        "mvs_upload_elapsed_ms": mvs_upload_elapsed,
        "mvs_upload_budget_ms": mvs_upload_budget,
        "mvs_upload_held_by_budget_cam_count": mvs_held,
        "mvs_upload_budget_overrun_rate": _as_float(_dig(compact, "global_bev.mvs_texture_upload.budget_overrun_rate")),
        "event_layer_display_budget_elapsed_ms": event_elapsed,
        "event_layer_display_budget_ms": event_budget,
        "event_layer_display_budget_held_cam_count": event_held,
        "event_layer_display_budget_overrun_rate": event_overrun_rate,
        "event_layer_display_budget_order": _dig(compact, "global_bev.event_layer_display_budget_order"),
        "event_layer_presentation_worker_enabled": pres_enabled,
        "event_layer_presentation_worker_mode": pres_mode,
        "event_layer_presentation_worker_state": pres_state,
        "event_layer_presentation_worker_worker_state": pres_worker_state,
        "event_layer_presentation_worker_payload_policy": pres_payload_policy,
        "event_layer_presentation_worker_read_points": pres_read_points,
        "event_layer_presentation_worker_compare_state": pres_compare,
        "event_layer_presentation_worker_mismatch_cam_count": pres_mismatch,
        "event_layer_presentation_worker_display_compare_state": pres_display_compare,
        "event_layer_presentation_worker_display_mismatch_cam_count": pres_display_mismatch,
        "event_layer_presentation_worker_delta_compare_state": pres_delta_compare,
        "event_layer_presentation_worker_delta_mismatch_cam_count": pres_delta_mismatch,
        "event_layer_presentation_worker_compare_lookup_state": pres_compare_lookup_state,
        "event_layer_presentation_worker_compare_target_delta_ticks": pres_compare_target_delta_ticks,
        "event_layer_presentation_worker_compare_tolerance_ticks": pres_compare_tolerance_ticks,
        "event_layer_presentation_worker_compare_exact_count": pres_compare_exact_count,
        "event_layer_presentation_worker_compare_near_count": pres_compare_near_count,
        "event_layer_presentation_worker_compare_miss_count": pres_compare_miss_count,
        "event_layer_presentation_worker_compare_hit_ratio": pres_compare_hit_ratio,
        "event_layer_presentation_worker_build_p95_ms": pres_build_p95,
        "event_layer_presentation_worker_fallback_count": pres_fallback,
        "event_layer_presentation_worker_fallback_reason": _dig(compact, "global_bev.event_layer_presentation_worker.fallback_reason"),
        "event_layer_presentation_worker_last_fallback_reason": _dig(compact, "global_bev.event_layer_presentation_worker.last_fallback_reason"),
        "event_layer_presentation_worker_active_consume_count": pres_active_consume_count,
        "event_layer_presentation_worker_active_consume_elapsed_ms": _as_float(_dig(compact, "global_bev.event_layer_presentation_worker.active_consume_elapsed_ms")),
        "event_layer_presentation_worker_bundle_hit_count": pres_bundle_hit_count,
        "event_layer_presentation_worker_bundle_miss_count": pres_bundle_miss_count,
        "event_layer_presentation_worker_bundle_hit_ratio": pres_bundle_hit_ratio,
        "event_layer_presentation_worker_consume_exact_count": pres_consume_exact_count,
        "event_layer_presentation_worker_consume_near_count": pres_consume_near_count,
        "event_layer_presentation_worker_bundle_ring_count": pres_bundle_ring_count,
        "event_layer_presentation_worker_bundle_ring_size": _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.bundle_ring_size")),
        "event_layer_presentation_worker_cached_target_min": _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.cached_target_min"), -1),
        "event_layer_presentation_worker_cached_target_max": _as_int(_dig(compact, "global_bev.event_layer_presentation_worker.cached_target_max"), -1),
        "gui_upload_selected_p95_ms": _as_float(_dig(compact, "global_bev.alignment.gui_upload_selected_p95_ms")),
        "gui_upload_event_layer_p95_ms": _as_float(_dig(compact, "global_bev.alignment.gui_upload_event_layer_p95_ms")),
        "gui_event_sparse_read_p95_ms": _as_float(_dig(compact, "global_bev.alignment.gui_event_sparse_read_p95_ms")),
        "gui_event_sparse_sync_gate_p95_ms": _as_float(_dig(compact, "global_bev.alignment.gui_event_sparse_sync_gate_p95_ms")),
        "gui_event_sparse_project_p95_ms": _as_float(_dig(compact, "global_bev.alignment.gui_event_sparse_project_p95_ms")),
        "gui_paint_total_p95_ms": _as_float(_dig(compact, "global_bev.alignment.gui_paint_total_p95_ms")),
        "global_yolo_inferred_fps_total": yolo_inferred_fps,
        "global_yolo_published_fps_total": yolo_published_fps,
        "global_yolo_async_record_postprocess_enabled": yolo_async,
        "global_yolo_postprocess_queue_depth": yolo_queue_depth,
        "global_yolo_postprocess_dropped_batches": yolo_dropped,
        "global_yolo_polygon_contract_ok": yolo_polygon_ok,
        "global_yolo_published_persons_with_polygon_ratio": yolo_person_polygon_ratio,
        "global_yolo_input_alignment_audit_enabled": yolo_input_align_enabled,
        "global_yolo_input_alignment_full_batch_ratio": yolo_input_align_full_ratio,
        "global_yolo_input_alignment_same_sync_batch_ratio": yolo_input_align_same_sync_ratio,
        "global_yolo_input_alignment_current_aligned_full_batch_ratio": yolo_input_align_current_full_ratio,
        "global_yolo_input_alignment_sync_full_sets_within_wait_ratio": yolo_input_align_within_wait_ratio,
        "global_yolo_input_alignment_sync_full_wait_ms_avg": yolo_input_align_wait_avg_ms,
        "global_yolo_input_alignment_visual_mid_spread_ms_max": yolo_input_align_visual_spread_max_ms,
        "global_yolo_input_alignment_visual_mid_spread_ms_p95": yolo_input_align_visual_spread_p95_ms,
        "global_yolo_input_alignment_exposure_spread_us_p95": yolo_input_align_exposure_spread_p95_us,
        "global_yolo_input_buffer_enabled": yolo_input_buffer_enabled,
        "global_yolo_input_buffer_aligned_full_ratio": yolo_input_buffer_aligned_full_ratio,
        "global_yolo_input_buffer_fallback_ratio": yolo_input_buffer_fallback_ratio,
    }
    return {
        "kind": "fullsys_node_assessment",
        "node": args.node,
        "created_wall_us": int(time.time() * 1_000_000),
        "overall": overall,
        "run_dir": compact.get("run_dir"),
        "archive": compact.get("archive"),
        "launcher_rc": compact.get("launcher_rc"),
        "runtime_overall_state": compact.get("runtime_overall_state"),
        "status_key_ok": compact.get("status_key_ok"),
        "metrics": metrics,
        "findings": findings,
        "thresholds": {
            "bev_render_warn_fps": args.bev_render_warn_fps,
            "bev_render_fail_fps": args.bev_render_fail_fps,
            "bev_output_warn_fps": args.bev_output_warn_fps,
            "required_mvs_synced_cams": args.required_mvs_synced_cams,
            "budget_overrun_warn_ratio": args.budget_overrun_warn_ratio,
            "yolo_published_warn_fps": args.yolo_published_warn_fps,
            "yolo_published_fail_fps": args.yolo_published_fail_fps,
            "yolo_queue_depth_warn": args.yolo_queue_depth_warn,
            "yolo_polygon_ratio_warn": args.yolo_polygon_ratio_warn,
        },
    }


def render_markdown(assessment: Dict[str, Any]) -> str:
    m = assessment.get("metrics", {})
    lines = [
        f"# Full-System Node Assessment: {assessment.get('node') or 'unnamed'}",
        "",
        f"- Overall: **{assessment.get('overall')}**",
        f"- Run dir: `{assessment.get('run_dir')}`",
        f"- Archive: `{assessment.get('archive')}`",
        f"- Launcher rc: `{assessment.get('launcher_rc')}`",
        f"- Runtime: `{assessment.get('runtime_overall_state')}`",
        f"- Status key audit: `{assessment.get('status_key_ok')}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "render_fps",
        "output_fps",
        "mvs_synced_cam_count",
        "event_sync_outlier_cam_count",
        "mvs_upload_elapsed_ms",
        "mvs_upload_budget_ms",
        "mvs_upload_held_by_budget_cam_count",
        "event_layer_display_budget_elapsed_ms",
        "event_layer_display_budget_ms",
        "event_layer_display_budget_held_cam_count",
        "event_layer_presentation_worker_enabled",
        "event_layer_presentation_worker_mode",
        "event_layer_presentation_worker_state",
        "event_layer_presentation_worker_worker_state",
        "event_layer_presentation_worker_payload_policy",
        "event_layer_presentation_worker_read_points",
        "event_layer_presentation_worker_compare_state",
        "event_layer_presentation_worker_mismatch_cam_count",
        "event_layer_presentation_worker_display_compare_state",
        "event_layer_presentation_worker_display_mismatch_cam_count",
        "event_layer_presentation_worker_delta_compare_state",
        "event_layer_presentation_worker_delta_mismatch_cam_count",
        "event_layer_presentation_worker_compare_lookup_state",
        "event_layer_presentation_worker_compare_target_delta_ticks",
        "event_layer_presentation_worker_compare_tolerance_ticks",
        "event_layer_presentation_worker_compare_exact_count",
        "event_layer_presentation_worker_compare_near_count",
        "event_layer_presentation_worker_compare_miss_count",
        "event_layer_presentation_worker_compare_hit_ratio",
        "event_layer_presentation_worker_build_p95_ms",
        "event_layer_presentation_worker_active_consume_count",
        "event_layer_presentation_worker_active_consume_elapsed_ms",
        "event_layer_presentation_worker_fallback_count",
        "event_layer_presentation_worker_fallback_reason",
        "event_layer_presentation_worker_last_fallback_reason",
        "event_layer_presentation_worker_bundle_hit_count",
        "event_layer_presentation_worker_bundle_miss_count",
        "event_layer_presentation_worker_bundle_hit_ratio",
        "event_layer_presentation_worker_consume_exact_count",
        "event_layer_presentation_worker_consume_near_count",
        "event_layer_presentation_worker_bundle_ring_count",
        "event_layer_presentation_worker_bundle_ring_size",
        "event_layer_presentation_worker_cached_target_min",
        "event_layer_presentation_worker_cached_target_max",
        "gui_upload_selected_p95_ms",
        "gui_upload_event_layer_p95_ms",
        "gui_paint_total_p95_ms",
        "global_yolo_inferred_fps_total",
        "global_yolo_published_fps_total",
        "global_yolo_async_record_postprocess_enabled",
        "global_yolo_postprocess_queue_depth",
        "global_yolo_postprocess_dropped_batches",
        "global_yolo_polygon_contract_ok",
        "global_yolo_published_persons_with_polygon_ratio",
        "global_yolo_input_alignment_audit_enabled",
        "global_yolo_input_alignment_full_batch_ratio",
        "global_yolo_input_alignment_same_sync_batch_ratio",
        "global_yolo_input_alignment_current_aligned_full_batch_ratio",
        "global_yolo_input_alignment_sync_full_sets_within_wait_ratio",
        "global_yolo_input_alignment_sync_full_wait_ms_avg",
        "global_yolo_input_alignment_visual_mid_spread_ms_max",
        "global_yolo_input_alignment_visual_mid_spread_ms_p95",
        "global_yolo_input_alignment_exposure_spread_us_p95",
        "global_yolo_input_buffer_enabled",
        "global_yolo_input_buffer_aligned_full_ratio",
        "global_yolo_input_buffer_fallback_ratio",
    ):
        lines.append(f"| `{key}` | `{m.get(key)}` |")
    lines += ["", "## Findings", ""]
    findings = assessment.get("findings", [])
    if not findings:
        lines.append("- PASS: no findings.")
    else:
        for item in findings:
            lines.append(f"- {item.get('level')}: `{item.get('key')}` - {item.get('message')}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a full-system test report against node-closure guardrails.")
    ap.add_argument("--input", required=True, help="Path to compact or full test report JSON.")
    ap.add_argument("--node", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--bev-render-warn-fps", type=float, default=29.0)
    ap.add_argument("--bev-render-fail-fps", type=float, default=25.0)
    ap.add_argument("--bev-output-warn-fps", type=float, default=8.0)
    ap.add_argument("--required-mvs-synced-cams", type=int, default=8)
    ap.add_argument("--budget-overrun-warn-ratio", type=float, default=1.1)
    ap.add_argument("--yolo-published-warn-fps", type=float, default=300.0)
    ap.add_argument("--yolo-published-fail-fps", type=float, default=260.0)
    ap.add_argument("--yolo-queue-depth-warn", type=int, default=1)
    ap.add_argument("--yolo-polygon-ratio-warn", type=float, default=0.98)
    ap.add_argument("--require-v24c6-keys", action="store_true", default=True)
    ap.add_argument("--no-require-v24c6-keys", dest="require_v24c6_keys", action="store_false")
    ap.add_argument("--strict-exit", action="store_true", help="Exit non-zero when assessment is FAIL.")
    args = ap.parse_args()

    report = _read_json(Path(args.input))
    assessment = evaluate(report, args)
    text = json.dumps(assessment, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_markdown(assessment), encoding="utf-8")
    print(text)
    if args.strict_exit and assessment.get("overall") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
