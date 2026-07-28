#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record YOLO-after-recognition person-ID datasets.

This sidecar intentionally consumes only structured outputs downstream of YOLO:
per-camera ``human_result_*.jsonl`` and ``global_person_tracks.jsonl``. It
sanitizes mask/segmentation fields before writing datasets so the result is
lightweight and suitable for Global Person ID regression work. For V19 anchor
validation it can optionally preserve polygon fields while still dropping heavy
mask bitmap/RLE payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple


MASK_KEY_TOKENS = (
    "mask",
    "segmentation",
    "contour",
    "polygon",
    "rle",
    "bitmap",
    "alpha_mask",
)

POLYGON_KEY_TOKENS = (
    "polygon",
)

HEAVY_MASK_KEY_TOKENS = (
    "rle",
    "bitmap",
    "alpha_mask",
    "mask_bitmap",
    "mask_rle",
)


def _now_us() -> int:
    return int(time.time() * 1_000_000)


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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def _parse_cams(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").replace(";", ",").split(",") if x.strip()]


def _slug(value: str, default: str = "dataset") -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or default


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(fp: TextIO, payload: Dict[str, Any]) -> None:
    fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            obj = {"value": obj}
        st = path.stat()
        obj["_path"] = str(path)
        obj["_mtime"] = st.st_mtime
        obj["_age_ms"] = max(0.0, (time.time() - st.st_mtime) * 1000.0)
        return obj
    except Exception as exc:
        return {"_path": str(path), "_error": f"{type(exc).__name__}: {exc}"}


def _stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _is_polygon_key(key: Any) -> bool:
    text = str(key or "").strip().lower()
    return any(token in text for token in POLYGON_KEY_TOKENS)


def _is_mask_key(key: Any, include_mask_polygons: bool = False) -> bool:
    text = str(key or "").strip().lower()
    if include_mask_polygons and _is_polygon_key(text):
        if not any(token in text for token in HEAVY_MASK_KEY_TOKENS):
            return False
    return any(token in text for token in MASK_KEY_TOKENS)


def _sanitize_no_mask(obj: Any, include_mask_polygons: bool = False) -> Tuple[Any, int]:
    removed = 0
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            if _is_mask_key(key, include_mask_polygons=include_mask_polygons):
                removed += 1
                continue
            cleaned, sub_removed = _sanitize_no_mask(value, include_mask_polygons=include_mask_polygons)
            removed += sub_removed
            out[key] = cleaned
        return out, removed
    if isinstance(obj, list):
        out_list: List[Any] = []
        for item in obj:
            cleaned, sub_removed = _sanitize_no_mask(item, include_mask_polygons=include_mask_polygons)
            removed += sub_removed
            out_list.append(cleaned)
        return out_list, removed
    return obj, 0


class JsonlTailer:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.fp: Optional[TextIO] = None
        self.pos = 0
        self.start_at_end_on_open = True
        self.last_error = ""
        self.bad_lines = 0
        self.good_lines = 0
        self.last_good_wall_us = 0

    def close(self) -> None:
        try:
            if self.fp is not None:
                self.fp.close()
        except Exception:
            pass
        self.fp = None

    def mark_boundary_now(self) -> None:
        self.close()
        self.last_error = ""
        if self.path.exists():
            try:
                self.fp = self.path.open("r", encoding="utf-8", errors="replace")
                self.fp.seek(0, os.SEEK_END)
                self.pos = self.fp.tell()
                self.start_at_end_on_open = False
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.close()
                self.pos = 0
                self.start_at_end_on_open = False
        else:
            self.pos = 0
            self.start_at_end_on_open = False

    def _open(self) -> bool:
        if self.fp is not None:
            return True
        if not self.path.exists():
            return False
        try:
            self.fp = self.path.open("r", encoding="utf-8", errors="replace")
            if self.start_at_end_on_open:
                self.fp.seek(0, os.SEEK_END)
            else:
                self.fp.seek(max(0, int(self.pos)), os.SEEK_SET)
            self.pos = self.fp.tell()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            return False

    def poll(self, max_lines: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._open():
            return out
        try:
            try:
                size = self.path.stat().st_size
                if size < self.pos:
                    self.close()
                    self.pos = 0
                    self.start_at_end_on_open = False
                    return out
            except Exception:
                pass
            for _ in range(max(1, int(max_lines))):
                start_pos = self.fp.tell() if self.fp is not None else self.pos
                line = self.fp.readline() if self.fp is not None else ""
                if not line:
                    break
                if not line.endswith("\n"):
                    if self.fp is not None:
                        self.fp.seek(start_pos, os.SEEK_SET)
                    self.pos = start_pos
                    break
                self.pos = self.fp.tell() if self.fp is not None else self.pos
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                        self.good_lines += 1
                        self.last_good_wall_us = _now_us()
                        self.last_error = ""
                except Exception as exc:
                    self.bad_lines += 1
                    self.last_error = f"json:{type(exc).__name__}: {exc}"
            return out
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            return out


class PersonIdDatasetRecorder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd().resolve()
        self.sync_root = Path(args.sync_root).expanduser()
        if not self.sync_root.is_absolute():
            self.sync_root = self.project_root / self.sync_root
        self.output_root = Path(args.output_root).expanduser()
        if not self.output_root.is_absolute():
            self.output_root = self.project_root / self.output_root
        self.evidence_output_root = Path(getattr(args, "evidence_output_root", "recordings/evidence_replay_datasets")).expanduser()
        if not self.evidence_output_root.is_absolute():
            self.evidence_output_root = self.project_root / self.evidence_output_root
        self.cams = _parse_cams(args.cams)
        self.running = True
        self.recording = False
        self.session_id = ""
        self.dataset_kind = "person_id"
        self.dataset_type = ""
        self.session_dir: Optional[Path] = None
        self.started_wall_us = 0
        self.stopped_wall_us = 0
        self.last_error = ""
        self.last_control_seq = -1
        self.last_command = ""
        self.last_write_wall_us = 0
        self.last_status_wall = 0.0
        self.last_sample_wall = 0.0
        self.last_zero_person_keep_us: Dict[str, int] = {}
        self.default_include_mask_polygons = bool(getattr(args, "include_mask_polygons", False))
        self.include_mask_polygons = bool(self.default_include_mask_polygons)
        self.include_bullet_trajectory = False
        self.include_hit_debug = False

        self.control_path = self._abs_sync(args.control_json)
        self.status_path = self._abs_sync(args.status_json)
        self.latest_path = self._abs_sync(args.latest_json)
        self.global_tracks_path = self._abs_sync(args.global_person_jsonl)
        self.operator_annotations_path = self._abs_sync(args.operator_annotations_jsonl)
        self.status_sources = {
            "global_person_latest": self._abs_sync(args.global_person_latest_json),
            "global_person_status": self._abs_sync(args.global_person_status_json),
            "global_yolo_status": self._abs_sync(args.global_yolo_status_json),
            "system_status_latest": self._abs_sync(args.system_status_json),
        }

        self.human_tailers: Dict[str, JsonlTailer] = {
            cam: JsonlTailer(self._human_path(cam)) for cam in self.cams
        }
        self.track_tailer = JsonlTailer(self.global_tracks_path)
        self.annotation_tailer = JsonlTailer(self.operator_annotations_path)
        self.evidence_tailers: Dict[str, JsonlTailer] = {
            key: JsonlTailer(self._abs_sync(src)) for key, src, _out, _counter in self._evidence_source_specs()
        }
        self.files: Dict[str, TextIO] = {}
        self.counts: Dict[str, Any] = self._new_counts()

    def _evidence_source_specs(self) -> List[Tuple[str, str, str, str]]:
        return [
            ("bullet_points", "overlay_bullet_point_all.jsonl", "bullets/overlay_bullet_point_all.window.jsonl", "bullet_point_rows"),
            ("bullet_events", "overlay_bullet_event_all.jsonl", "bullets/overlay_bullet_event_all.window.jsonl", "bullet_event_rows"),
            ("pseudo_hits", "pseudo_hit_all.jsonl", "hit_judge/pseudo_hit_all.window.jsonl", "pseudo_hit_rows"),
            ("final_hits", "hit_candidate_all.jsonl", "hit_judge/hit_candidate_all.window.jsonl", "final_hit_rows"),
            ("strong_reasoning_shadow", "hit_candidate_shadow_strong_reasoning_all.jsonl", "hit_judge/hit_candidate_shadow_strong_reasoning_all.window.jsonl", "shadow_hit_rows"),
            ("reject_debug", "hit_reject_debug_all.jsonl", "hit_judge/hit_reject_debug_all.window.jsonl", "hit_reject_rows"),
            ("damage_events", "damage_attribution_events.jsonl", "hit_judge/damage_attribution_events.window.jsonl", "damage_event_rows"),
            ("bridge_sent", "game_event_bridge_sent.jsonl", "hit_judge/game_event_bridge_sent.window.jsonl", "bridge_sent_rows"),
        ]

    def _abs_sync(self, raw: str) -> Path:
        path = Path(str(raw or "")).expanduser()
        if not path.is_absolute():
            parts = path.parts
            if parts and parts[0] == self.sync_root.name:
                path = self.project_root / path
            else:
                path = self.sync_root / path
        return path

    def _human_path(self, cam: str) -> Path:
        template = str(self.args.human_jsonl_template or "{root}/human_result_{camera}.jsonl")
        text = template.format(root=str(self.sync_root), camera=cam, cam=cam, id=cam)
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _new_counts(self) -> Dict[str, Any]:
        return {
            "human_rows_total": 0,
            "human_person_rows_total": 0,
            "human_zero_person_rows_total": 0,
            "human_zero_person_skipped_total": 0,
            "global_person_track_rows": 0,
            "status_samples": 0,
            "annotation_rows": 0,
            "bullet_point_rows": 0,
            "bullet_event_rows": 0,
            "pseudo_hit_rows": 0,
            "final_hit_rows": 0,
            "shadow_hit_rows": 0,
            "hit_reject_rows": 0,
            "damage_event_rows": 0,
            "bridge_sent_rows": 0,
            "sanitized_mask_key_count": 0,
            "bad_lines": 0,
            "per_camera": {
                cam: {
                    "human_rows": 0,
                    "person_rows": 0,
                    "zero_person_rows": 0,
                    "zero_person_skipped": 0,
                    "sanitized_mask_key_count": 0,
                    "last_sync_index": -1,
                    "last_frame_id": -1,
                    "last_record_age_ms": -1.0,
                }
                for cam in self.cams
            },
        }

    def _excluded_key_tokens(self) -> List[str]:
        if not self.include_mask_polygons:
            return list(MASK_KEY_TOKENS)
        return [token for token in MASK_KEY_TOKENS if token not in POLYGON_KEY_TOKENS]

    def _human_output_name(self, cam: str) -> str:
        stem = "human_result_polygon" if self.include_mask_polygons else "human_result_sanitized"
        return f"{stem}_{cam}.jsonl"

    def _session_output_root(self) -> Path:
        if str(getattr(self, "dataset_kind", "person_id") or "person_id") == "evidence_replay":
            return self.evidence_output_root
        return self.output_root

    def _experiment_context(self) -> Dict[str, Any]:
        exp = _read_json(self.sync_root / "experiment_control.json")
        if not exp or exp.get("_error"):
            exp = {}
        context = {
            "run_id": os.environ.get("MRG_RUN_ID", ""),
            "experiment_id": str(exp.get("experiment_id", "")),
            "stage": str(exp.get("stage", "")),
            "config_hash": str(exp.get("config_hash", "")),
            "final_hit_authority": str((exp.get("hit_payload", {}) if isinstance(exp.get("hit_payload"), dict) else {}).get("final_hit_authority", "")),
            "global_person_target": str(exp.get("global_person_target", "")),
            "hit_profile_path": str(exp.get("hit_profile_path", "")),
            "global_id_profile_path": str(exp.get("global_id_profile_path", "")),
        }
        context["context_hash"] = _stable_hash(context)
        return context

    def _manifest(self, dataset_type: str, label: str, description: str) -> Dict[str, Any]:
        human_template = "human_result_polygon_{camera}.jsonl" if self.include_mask_polygons else "human_result_sanitized_{camera}.jsonl"
        if self.dataset_kind == "evidence_replay":
            human_template = "human/" + human_template
        experiment = self._experiment_context()
        return {
            "schema_version": 1,
            "kind": "person_id_dataset_manifest",
            "session_id": self.session_id,
            "dataset_kind": self.dataset_kind,
            "dataset_type": dataset_type,
            "label": label,
            "description": description,
            "created_wall_us": self.started_wall_us,
            "run_id": experiment.get("run_id", ""),
            "experiment_id": experiment.get("experiment_id", ""),
            "experiment_stage": experiment.get("stage", ""),
            "experiment_config_hash": experiment.get("config_hash", ""),
            "dataset_config_hash": experiment.get("context_hash", ""),
            "project_root": str(self.project_root),
            "sync_root": str(self.sync_root),
            "contains_mask_data": bool(self.include_mask_polygons),
            "contains_polygon_data": bool(self.include_mask_polygons),
            "include_mask_polygons": bool(self.include_mask_polygons),
            "include_bullet_trajectory": bool(self.include_bullet_trajectory),
            "include_hit_debug": bool(self.include_hit_debug),
            "source_stage": "after_yolo_recognition",
            "intended_use": "Evidence Replay regression for Global Person ID, bullet trajectory, hit judge, and realtime-like review",
            "not_intended_for": "YOLO/Event raw perception performance validation",
            "id_semantics": "global_id/global_player_id comes from Global Person ID output; virtual shooter id 99 is not a real person track.",
            "replay_contract": "Replay should consume this folder as a timestamped evidence source and preserve the original timing order where possible.",
            "excluded_key_tokens": self._excluded_key_tokens(),
            "preserved_key_tokens": list(POLYGON_KEY_TOKENS) if self.include_mask_polygons else [],
            "zero_person_keep_ms": float(getattr(self.args, "zero_person_keep_ms", 0.0) or 0.0),
            "experiment_context": experiment,
            "files": {
                "human_result_template": human_template,
                "global_person_tracks_baseline": "global_person_tracks_baseline.jsonl",
                "status_samples": "status_samples.jsonl",
                "operator_annotations": "operator_annotations.jsonl",
                "summary": "summary.json",
                "bullet_points": "bullets/overlay_bullet_point_all.window.jsonl" if self.include_bullet_trajectory else "",
                "bullet_events": "bullets/overlay_bullet_event_all.window.jsonl" if self.include_bullet_trajectory else "",
                "pseudo_hits": "hit_judge/pseudo_hit_all.window.jsonl" if self.include_hit_debug else "",
                "final_hits": "hit_judge/hit_candidate_all.window.jsonl" if self.include_hit_debug else "",
                "shadow_hits": "hit_judge/hit_candidate_shadow_strong_reasoning_all.window.jsonl" if self.include_hit_debug else "",
                "timeline_index": "index/timeline_index.json" if self.dataset_kind == "evidence_replay" else "",
            },
            "sources": {
                "human_result_template": str(self.args.human_jsonl_template),
                "global_person_tracks": str(self.global_tracks_path),
                "operator_annotations": str(self.operator_annotations_path),
                "status_sources": {k: str(v) for k, v in self.status_sources.items()},
            },
            "dataset_type_contract": {
                "negative": "No real person should be present; useful for false-track and stale-ID checks.",
                "positive": "Real person movement/ID continuity sample; annotate expected identity transitions.",
                "mixed": "Contains controlled positive and negative periods; use operator annotations to segment.",
            },
        }

    def _write_dataset_card(self, dataset_type: str, label: str, description: str) -> None:
        if self.session_dir is None:
            return
        human_template = "human_result_polygon_<cam>.jsonl" if self.include_mask_polygons else "human_result_sanitized_<cam>.jsonl"
        if self.dataset_kind == "evidence_replay":
            human_template = "human/" + human_template
        mask_line = "true" if self.include_mask_polygons else "false"
        polygon_line = "true" if self.include_mask_polygons else "false"
        evidence_line = (
            "Polygon fields are preserved for V19 anchor validation; heavy mask bitmap/RLE fields are removed."
            if self.include_mask_polygons
            else "Mask, polygon, segmentation, contour, RLE, bitmap, and alpha fields are removed."
        )
        lines = [
            "# Person ID Dataset",
            "",
            f"- session_id: `{self.session_id}`",
            f"- dataset_type: `{dataset_type}`",
            f"- dataset_kind: `{self.dataset_kind}`",
            f"- label: `{label}`",
            f"- source_stage: `after_yolo_recognition`",
            f"- contains_mask_data: `{mask_line}`",
            f"- contains_polygon_data: `{polygon_line}`",
            f"- include_bullet_trajectory: `{str(bool(self.include_bullet_trajectory)).lower()}`",
            f"- include_hit_debug: `{str(bool(self.include_hit_debug)).lower()}`",
            "",
            "This dataset records structured YOLO/person-ID evidence only.",
            evidence_line,
            "",
            "Files:",
            f"- `{human_template}`: per-camera YOLO person rows.",
            "- `global_person_tracks_baseline.jsonl`: current Global Person ID output for baseline comparison.",
            "- `status_samples.jsonl`: periodic status snapshots for runtime context.",
            "- `operator_annotations.jsonl`: operator labels copied from the status window annotation stream.",
            "- `bullets/*.window.jsonl`: full bullet trajectory/event streams, only for Evidence Replay sessions.",
            "- `hit_judge/*.window.jsonl`: pseudo/final/shadow/reject evidence, only for Evidence Replay sessions.",
            "- `index/*.json`: lightweight local-player/replay indexes generated at stop time.",
            "- `summary.json`: closed-session counters and risk hints.",
            "",
            f"Description: {description or '-'}",
            "",
        ]
        (self.session_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    def _open_session_files(self) -> None:
        if self.session_dir is None:
            raise RuntimeError("session_dir is not set")
        self.files = {}
        human_dir = self.session_dir / "human" if self.dataset_kind == "evidence_replay" else self.session_dir
        global_dir = self.session_dir / "global" if self.dataset_kind == "evidence_replay" else self.session_dir
        human_dir.mkdir(parents=True, exist_ok=True)
        global_dir.mkdir(parents=True, exist_ok=True)
        for cam in self.cams:
            path = human_dir / self._human_output_name(cam)
            self.files[f"human:{cam}"] = path.open("a", encoding="utf-8", buffering=1)
        self.files["tracks"] = (global_dir / "global_person_tracks_baseline.jsonl").open("a", encoding="utf-8", buffering=1)
        self.files["status"] = (self.session_dir / "status_samples.jsonl").open("a", encoding="utf-8", buffering=1)
        self.files["annotations"] = (self.session_dir / "operator_annotations.jsonl").open("a", encoding="utf-8", buffering=1)
        if self.dataset_kind == "evidence_replay":
            for key, _src, out_rel, _counter in self._evidence_source_specs():
                if key.startswith("bullet_") and not self.include_bullet_trajectory:
                    continue
                if (not key.startswith("bullet_")) and not self.include_hit_debug:
                    continue
                out_path = self.session_dir / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                self.files[f"evidence:{key}"] = out_path.open("a", encoding="utf-8", buffering=1)

    def _close_session_files(self) -> None:
        for fp in list(self.files.values()):
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass
        self.files = {}

    def _reset_tailers_to_now(self) -> None:
        for tailer in self.human_tailers.values():
            tailer.mark_boundary_now()
        self.track_tailer.mark_boundary_now()
        self.annotation_tailer.mark_boundary_now()
        for tailer in self.evidence_tailers.values():
            tailer.mark_boundary_now()

    def start_session(
        self,
        dataset_type: str,
        label: str,
        description: str,
        dataset_kind: str = "person_id",
        session_id: str = "",
        include_mask_polygons: Optional[bool] = None,
        include_bullet_trajectory: Optional[bool] = None,
        include_hit_debug: Optional[bool] = None,
    ) -> None:
        dataset_kind = str(dataset_kind or "person_id").strip().lower()
        if dataset_kind in {"evidence", "evidence_replay_dataset", "replay"}:
            dataset_kind = "evidence_replay"
        if dataset_kind not in {"person_id", "evidence_replay"}:
            dataset_kind = "person_id"
        dataset_type = str(dataset_type or "mixed").strip().lower()
        if dataset_type in {"no_person", "empty", "idle"}:
            dataset_type = "negative"
        if dataset_type not in {"negative", "positive", "mixed", "battle_1v1"}:
            dataset_type = "mixed"
        if self.recording:
            self.stop_session(reason="rotated_by_new_start")
        self.started_wall_us = _now_us()
        self.stopped_wall_us = 0
        self.dataset_kind = dataset_kind
        if include_mask_polygons is None:
            include_mask_polygons = bool(self.default_include_mask_polygons or dataset_kind == "evidence_replay")
        if include_bullet_trajectory is None:
            include_bullet_trajectory = bool(dataset_kind == "evidence_replay")
        if include_hit_debug is None:
            include_hit_debug = bool(dataset_kind == "evidence_replay")
        self.include_mask_polygons = bool(include_mask_polygons)
        self.include_bullet_trajectory = bool(include_bullet_trajectory and dataset_kind == "evidence_replay")
        self.include_hit_debug = bool(include_hit_debug and dataset_kind == "evidence_replay")
        kind_slug = "evidence" if dataset_kind == "evidence_replay" else "person"
        session_override = _slug(str(session_id or ""), "")
        self.session_id = session_override or f"{time.strftime('%Y%m%d_%H%M%S')}_{kind_slug}_{dataset_type}_{_slug(label, 'sample')}"
        self.dataset_type = dataset_type
        self.session_dir = self._session_output_root() / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.counts = self._new_counts()
        self.last_zero_person_keep_us = {}
        self._reset_tailers_to_now()
        self._open_session_files()
        manifest = self._manifest(dataset_type, label, description)
        _atomic_write_json(self.session_dir / "manifest.json", manifest)
        self._write_dataset_card(dataset_type, label, description)
        self.recording = True
        self.last_error = ""
        self.last_write_wall_us = self.started_wall_us
        self._write_status(force=True)
        print(f"[person_id_dataset] start session={self.session_id} dir={self.session_dir}", flush=True)

    def stop_session(self, reason: str = "stop_command") -> None:
        if not self.recording and self.session_dir is None:
            return
        self._drain_sources()
        self.stopped_wall_us = _now_us()
        summary = self._summary(reason)
        if self.session_dir is not None:
            _atomic_write_json(self.session_dir / "summary.json", summary)
            self._write_indexes(summary)
        self._close_session_files()
        self.recording = False
        self._write_status(force=True)
        self._write_latest(summary)
        print(f"[person_id_dataset] stop session={self.session_id} reason={reason}", flush=True)

    def _summary(self, reason: str = "") -> Dict[str, Any]:
        duration_ms = 0.0
        if self.started_wall_us:
            end_us = self.stopped_wall_us or _now_us()
            duration_ms = max(0.0, (end_us - self.started_wall_us) / 1000.0)
        experiment = self._experiment_context()
        return {
            "schema_version": 1,
            "kind": "person_id_dataset_summary",
            "state": "RECORDING" if self.recording else "READY",
            "recording": bool(self.recording),
            "dataset_kind": self.dataset_kind,
            "session_id": self.session_id,
            "dataset_type": self.dataset_type,
            "session_dir": str(self.session_dir or ""),
            "run_id": experiment.get("run_id", ""),
            "experiment_id": experiment.get("experiment_id", ""),
            "experiment_stage": experiment.get("stage", ""),
            "experiment_config_hash": experiment.get("config_hash", ""),
            "dataset_config_hash": experiment.get("context_hash", ""),
            "started_wall_us": self.started_wall_us,
            "stopped_wall_us": self.stopped_wall_us,
            "duration_ms": duration_ms,
            "stop_reason": reason,
            "contains_mask_data": bool(self.include_mask_polygons),
            "contains_polygon_data": bool(self.include_mask_polygons),
            "include_mask_polygons": bool(self.include_mask_polygons),
            "include_bullet_trajectory": bool(self.include_bullet_trajectory),
            "include_hit_debug": bool(self.include_hit_debug),
            "counts": self.counts,
            "experiment_context": experiment,
            "last_error": self.last_error,
        }

    def _write_latest(self, payload: Dict[str, Any]) -> None:
        latest = dict(payload)
        latest["kind"] = "person_id_dataset_latest"
        latest["updated_wall_us"] = _now_us()
        _atomic_write_json(self.latest_path, latest)

    def _control(self) -> Dict[str, Any]:
        return _read_json(self.control_path)

    def _handle_control(self) -> None:
        cmd_obj = self._control()
        seq = _safe_int(cmd_obj.get("seq"), -1)
        if seq < 0 or seq == self.last_control_seq:
            return
        self.last_control_seq = seq
        cmd = str(cmd_obj.get("cmd", cmd_obj.get("command", "")) or "").strip().lower()
        self.last_command = cmd
        try:
            if cmd == "start":
                dataset_kind = str(cmd_obj.get("dataset_kind", cmd_obj.get("kind", "person_id")) or "person_id")
                include_polygons = cmd_obj.get("include_mask_polygons", None)
                include_bullets = cmd_obj.get("include_bullet_trajectory", None)
                include_hit_debug = cmd_obj.get("include_hit_debug", None)
                self.start_session(
                    str(cmd_obj.get("dataset_type", "mixed") or "mixed"),
                    str(cmd_obj.get("label", cmd_obj.get("name", "")) or ""),
                    str(cmd_obj.get("description", "") or ""),
                    dataset_kind=dataset_kind,
                    session_id=str(cmd_obj.get("session_id", "") or ""),
                    include_mask_polygons=_as_bool(include_polygons, False) if include_polygons is not None else None,
                    include_bullet_trajectory=_as_bool(include_bullets, False) if include_bullets is not None else None,
                    include_hit_debug=_as_bool(include_hit_debug, False) if include_hit_debug is not None else None,
                )
            elif cmd == "stop":
                self.stop_session(reason="stop_command")
            elif cmd:
                self.last_error = f"unknown command: {cmd}"
        except Exception as exc:
            self.last_error = f"control:{type(exc).__name__}: {exc}"
            self.recording = False
            self._close_session_files()
            print(f"[person_id_dataset][ERROR] {self.last_error}", flush=True)
        self._write_status(force=True)

    def _record_human_rows(self) -> None:
        for cam, tailer in self.human_tailers.items():
            rows = tailer.poll(int(self.args.max_lines_per_poll))
            if not rows:
                continue
            fp = self.files.get(f"human:{cam}")
            if fp is None:
                continue
            cam_counts = self.counts["per_camera"].setdefault(cam, {})
            for row in rows:
                person_count_raw = _safe_int(row.get("person_count"), 0)
                now_us = _now_us()
                if person_count_raw <= 0:
                    keep_ms = max(0.0, float(getattr(self.args, "zero_person_keep_ms", 0.0) or 0.0))
                    last_keep = int(self.last_zero_person_keep_us.get(cam, 0) or 0)
                    if keep_ms > 0.0 and last_keep > 0 and (now_us - last_keep) < int(keep_ms * 1000.0):
                        self.counts["human_zero_person_skipped_total"] += 1
                        cam_counts["zero_person_skipped"] = _safe_int(cam_counts.get("zero_person_skipped"), 0) + 1
                        continue
                    self.last_zero_person_keep_us[cam] = now_us
                cleaned, removed = _sanitize_no_mask(row, include_mask_polygons=self.include_mask_polygons)
                if not isinstance(cleaned, dict):
                    continue
                cleaned["dataset_ingest_wall_us"] = now_us
                cleaned["dataset_source_file"] = str(tailer.path)
                _append_jsonl(fp, cleaned)
                self.counts["human_rows_total"] += 1
                cam_counts["human_rows"] = _safe_int(cam_counts.get("human_rows"), 0) + 1
                person_count = _safe_int(cleaned.get("person_count"), 0)
                if person_count <= 0:
                    self.counts["human_zero_person_rows_total"] += 1
                    cam_counts["zero_person_rows"] = _safe_int(cam_counts.get("zero_person_rows"), 0) + 1
                else:
                    self.counts["human_person_rows_total"] += person_count
                    cam_counts["person_rows"] = _safe_int(cam_counts.get("person_rows"), 0) + person_count
                self.counts["sanitized_mask_key_count"] += removed
                cam_counts["sanitized_mask_key_count"] = _safe_int(cam_counts.get("sanitized_mask_key_count"), 0) + removed
                cam_counts["last_sync_index"] = _safe_int(cleaned.get("sync_index", cleaned.get("sync_index_hint")), -1)
                cam_counts["last_frame_id"] = _safe_int(cleaned.get("frame_id_local", cleaned.get("frame_id")), -1)
                capture_wall_us = _safe_int(cleaned.get("capture_wall_us"), 0)
                cam_counts["last_record_age_ms"] = max(0.0, (_now_us() - capture_wall_us) / 1000.0) if capture_wall_us > 0 else -1.0
                self.last_write_wall_us = _now_us()

    def _record_tracks(self) -> None:
        rows = self.track_tailer.poll(int(self.args.max_lines_per_poll))
        if not rows:
            return
        fp = self.files.get("tracks")
        if fp is None:
            return
        for row in rows:
            cleaned, removed = _sanitize_no_mask(row, include_mask_polygons=self.include_mask_polygons)
            if not isinstance(cleaned, dict):
                continue
            cleaned["dataset_ingest_wall_us"] = _now_us()
            cleaned["dataset_source_file"] = str(self.track_tailer.path)
            _append_jsonl(fp, cleaned)
            self.counts["global_person_track_rows"] += 1
            self.counts["sanitized_mask_key_count"] += removed
            self.last_write_wall_us = _now_us()

    def _record_annotations(self) -> None:
        rows = self.annotation_tailer.poll(int(self.args.max_lines_per_poll))
        if not rows:
            return
        fp = self.files.get("annotations")
        if fp is None:
            return
        for row in rows:
            cleaned, removed = _sanitize_no_mask(row, include_mask_polygons=self.include_mask_polygons)
            if not isinstance(cleaned, dict):
                continue
            cleaned["dataset_ingest_wall_us"] = _now_us()
            _append_jsonl(fp, cleaned)
            self.counts["annotation_rows"] += 1
            self.counts["sanitized_mask_key_count"] += removed
            self.last_write_wall_us = _now_us()

    def _record_evidence_rows(self) -> None:
        if self.dataset_kind != "evidence_replay":
            return
        max_lines = int(getattr(self.args, "evidence_max_lines_per_poll", 4096) or 4096)
        for key, _src, _out_rel, counter_key in self._evidence_source_specs():
            if key.startswith("bullet_") and not self.include_bullet_trajectory:
                continue
            if (not key.startswith("bullet_")) and not self.include_hit_debug:
                continue
            tailer = self.evidence_tailers.get(key)
            fp = self.files.get(f"evidence:{key}")
            if tailer is None or fp is None:
                continue
            rows = tailer.poll(max_lines)
            if not rows:
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                out = dict(row)
                out["dataset_ingest_wall_us"] = _now_us()
                out["dataset_source_file"] = str(tailer.path)
                _append_jsonl(fp, out)
                self.counts[counter_key] = _safe_int(self.counts.get(counter_key), 0) + 1
                self.last_write_wall_us = _now_us()

    def _record_status_sample(self, force: bool = False) -> None:
        now = time.time()
        interval = max(0.1, float(self.args.sample_status_interval_ms) / 1000.0)
        if not force and now - self.last_sample_wall < interval:
            return
        fp = self.files.get("status")
        if fp is None:
            return
        sample = {
            "schema_version": 1,
            "kind": "person_id_dataset_status_sample",
            "wall_us": _now_us(),
            "session_id": self.session_id,
            "dataset_type": self.dataset_type,
            "sources": {},
        }
        for name, path in self.status_sources.items():
            obj, removed = _sanitize_no_mask(_read_json(path), include_mask_polygons=self.include_mask_polygons)
            sample["sources"][name] = obj
            self.counts["sanitized_mask_key_count"] += removed
        _append_jsonl(fp, sample)
        self.counts["status_samples"] += 1
        self.last_sample_wall = now
        self.last_write_wall_us = _now_us()

    def _poll_sources(self) -> None:
        if not self.recording:
            return
        self._record_human_rows()
        self._record_tracks()
        self._record_annotations()
        self._record_evidence_rows()
        self._record_status_sample()
        bad = self.track_tailer.bad_lines + self.annotation_tailer.bad_lines
        bad += sum(t.bad_lines for t in self.human_tailers.values())
        bad += sum(t.bad_lines for t in self.evidence_tailers.values())
        self.counts["bad_lines"] = bad

    def _total_recorded_rows(self) -> int:
        total = 0
        for key, value in self.counts.items():
            if key == "per_camera":
                continue
            if key.endswith("_rows") or key.endswith("_rows_total") or key.endswith("_samples"):
                total += _safe_int(value, 0)
        return total

    def _drain_sources(self) -> None:
        for _ in range(max(1, int(getattr(self.args, "stop_drain_rounds", 20) or 20))):
            before = self._total_recorded_rows()
            self._poll_sources()
            after = self._total_recorded_rows()
            if after <= before:
                break

    def _write_indexes(self, summary: Dict[str, Any]) -> None:
        if self.session_dir is None or self.dataset_kind != "evidence_replay":
            return
        index_dir = self.session_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        files: Dict[str, Any] = {}
        for path in sorted(self.session_dir.rglob("*.jsonl")):
            try:
                rel = path.relative_to(self.session_dir).as_posix()
                st = path.stat()
                files[rel] = {"size": st.st_size, "mtime": st.st_mtime}
            except Exception:
                continue
        timeline = {
            "schema_version": 1,
            "kind": "evidence_replay_timeline_index",
            "index_kind": "file_inventory_v1",
            "session_id": self.session_id,
            "dataset_kind": self.dataset_kind,
            "started_wall_us": self.started_wall_us,
            "stopped_wall_us": self.stopped_wall_us,
            "counts": self.counts,
            "files": files,
            "summary": {
                "duration_ms": summary.get("duration_ms"),
                "stop_reason": summary.get("stop_reason"),
                "include_mask_polygons": self.include_mask_polygons,
                "include_bullet_trajectory": self.include_bullet_trajectory,
                "include_hit_debug": self.include_hit_debug,
            },
        }
        camera_index = {
            "schema_version": 1,
            "kind": "evidence_replay_camera_index",
            "session_id": self.session_id,
            "cameras": {
                cam: {
                    "human_file": f"human/{self._human_output_name(cam)}",
                    "counts": (self.counts.get("per_camera", {}) if isinstance(self.counts.get("per_camera"), dict) else {}).get(cam, {}),
                }
                for cam in self.cams
            },
        }
        bullet_index = {
            "schema_version": 1,
            "kind": "evidence_replay_bullet_track_index",
            "index_kind": "source_file_inventory_v1",
            "session_id": self.session_id,
            "bullet_point_file": "bullets/overlay_bullet_point_all.window.jsonl",
            "bullet_event_file": "bullets/overlay_bullet_event_all.window.jsonl",
            "bullet_point_rows": _safe_int(self.counts.get("bullet_point_rows"), 0),
            "bullet_event_rows": _safe_int(self.counts.get("bullet_event_rows"), 0),
        }
        _atomic_write_json(index_dir / "timeline_index.json", timeline)
        _atomic_write_json(index_dir / "camera_index.json", camera_index)
        _atomic_write_json(index_dir / "bullet_track_index.json", bullet_index)

    def _source_status(self) -> Dict[str, Any]:
        per_cam: Dict[str, Any] = {}
        for cam, tailer in self.human_tailers.items():
            try:
                st = tailer.path.stat()
                exists = True
                age_ms = max(0.0, (time.time() - st.st_mtime) * 1000.0)
                size = st.st_size
            except Exception:
                exists = False
                age_ms = -1.0
                size = 0
            per_cam[cam] = {
                "path": str(tailer.path),
                "exists": exists,
                "age_ms": age_ms,
                "size": size,
                "last_error": tailer.last_error,
                "good_lines": tailer.good_lines,
                "bad_lines": tailer.bad_lines,
            }
        evidence_status: Dict[str, Any] = {}
        for key, _src, _out_rel, counter_key in self._evidence_source_specs():
            tailer = self.evidence_tailers.get(key)
            if tailer is None:
                continue
            try:
                st = tailer.path.stat()
                exists = True
                age_ms = max(0.0, (time.time() - st.st_mtime) * 1000.0)
                size = st.st_size
            except Exception:
                exists = False
                age_ms = -1.0
                size = 0
            evidence_status[key] = {
                "path": str(tailer.path),
                "exists": exists,
                "age_ms": age_ms,
                "size": size,
                "counter_key": counter_key,
                "rows": _safe_int(self.counts.get(counter_key), 0),
                "last_error": tailer.last_error,
                "good_lines": tailer.good_lines,
                "bad_lines": tailer.bad_lines,
            }
        return {
            "human_result": per_cam,
            "global_person_tracks": {
                "path": str(self.global_tracks_path),
                "exists": self.global_tracks_path.exists(),
                "age_ms": _read_json(self.global_tracks_path).get("_age_ms") if self.global_tracks_path.suffix == ".json" else (
                    max(0.0, (time.time() - self.global_tracks_path.stat().st_mtime) * 1000.0)
                    if self.global_tracks_path.exists() else -1.0
                ),
                "last_error": self.track_tailer.last_error,
            },
            "operator_annotations": {
                "path": str(self.operator_annotations_path),
                "exists": self.operator_annotations_path.exists(),
                "last_error": self.annotation_tailer.last_error,
            },
            "evidence_replay": evidence_status,
        }

    def _write_status(self, force: bool = False) -> None:
        now = time.time()
        interval = max(0.1, float(self.args.status_interval_ms) / 1000.0)
        if not force and now - self.last_status_wall < interval:
            return
        last_write_age_ms = -1.0
        if self.last_write_wall_us > 0:
            last_write_age_ms = max(0.0, (_now_us() - self.last_write_wall_us) / 1000.0)
        state = "RECORDING" if self.recording else "READY"
        if self.last_error:
            state = "ERROR" if not self.recording else "RECORDING_DEGRADED"
        experiment = self._experiment_context()
        payload = {
            "schema_version": 1,
            "kind": "person_id_dataset_status",
            "title_cn": "人员ID数据集",
            "state": state,
            "recording": bool(self.recording),
            "dataset_kind": self.dataset_kind,
            "dataset_type": self.dataset_type,
            "session_id": self.session_id,
            "session_dir": str(self.session_dir or ""),
            "run_id": experiment.get("run_id", ""),
            "experiment_id": experiment.get("experiment_id", ""),
            "experiment_stage": experiment.get("stage", ""),
            "experiment_config_hash": experiment.get("config_hash", ""),
            "dataset_config_hash": experiment.get("context_hash", ""),
            "output_root": str(self._session_output_root()),
            "contains_mask_data": bool(self.include_mask_polygons),
            "contains_polygon_data": bool(self.include_mask_polygons),
            "include_mask_polygons": bool(self.include_mask_polygons),
            "include_bullet_trajectory": bool(self.include_bullet_trajectory),
            "include_hit_debug": bool(self.include_hit_debug),
            "source_stage": "after_yolo_recognition",
            "zero_person_keep_ms": float(getattr(self.args, "zero_person_keep_ms", 0.0) or 0.0),
            "control_file": str(self.control_path),
            "control_seq": self.last_control_seq,
            "last_command": self.last_command,
            "last_error": self.last_error,
            "last_write_age_ms": last_write_age_ms,
            "updated_wall_us": _now_us(),
            "counts": self.counts,
            "experiment_context": experiment,
            "sources": self._source_status(),
        }
        _atomic_write_json(self.status_path, payload)
        self.last_status_wall = now

    def loop(self) -> int:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.evidence_output_root.mkdir(parents=True, exist_ok=True)
        self._write_status(force=True)
        print(f"[person_id_dataset] ready control={self.control_path} output_root={self.output_root} evidence_output_root={self.evidence_output_root}", flush=True)
        while self.running:
            self._handle_control()
            self._poll_sources()
            self._write_status()
            time.sleep(max(0.01, float(self.args.poll_ms) / 1000.0))
        if self.recording:
            self.stop_session(reason="process_exit")
        self._write_status(force=True)
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="YOLO-after-recognition Person ID dataset recorder.")
    ap.add_argument("--project-root", default="")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--cams", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--human-jsonl-template", default="{root}/human_result_{camera}.jsonl")
    ap.add_argument("--global-person-jsonl", default="global_person_tracks.jsonl")
    ap.add_argument("--global-person-latest-json", default="global_person_latest.json")
    ap.add_argument("--global-person-status-json", default="global_person_status.json")
    ap.add_argument("--global-yolo-status-json", default="global_yolo_status.json")
    ap.add_argument("--system-status-json", default="system_status_latest.json")
    ap.add_argument("--operator-annotations-jsonl", default="operator_annotations.jsonl")
    ap.add_argument("--control-json", default="person_id_dataset_control.json")
    ap.add_argument("--status-json", default="person_id_dataset_status.json")
    ap.add_argument("--latest-json", default="person_id_dataset_latest.json")
    ap.add_argument("--output-root", default="recordings/person_id_datasets")
    ap.add_argument("--evidence-output-root", default="recordings/evidence_replay_datasets")
    ap.add_argument("--poll-ms", type=float, default=100.0)
    ap.add_argument("--status-interval-ms", type=float, default=1000.0)
    ap.add_argument("--sample-status-interval-ms", type=float, default=1000.0)
    ap.add_argument("--max-lines-per-poll", type=int, default=512)
    ap.add_argument("--evidence-max-lines-per-poll", type=int, default=4096)
    ap.add_argument("--stop-drain-rounds", type=int, default=20)
    ap.add_argument("--zero-person-keep-ms", type=float, default=0.0,
                    help="When >0, keep at most one zero-person human_result row per camera per interval. Person rows are never downsampled.")
    ap.add_argument("--include-mask-polygons", action="store_true", default=False,
                    help="Preserve polygon fields such as mask_polygons while still dropping heavy mask bitmap/RLE fields.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    recorder = PersonIdDatasetRecorder(args)

    def _stop(_signum: int, _frame: Any) -> None:
        recorder.running = False

    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass
    return recorder.loop()


if __name__ == "__main__":
    raise SystemExit(main())
