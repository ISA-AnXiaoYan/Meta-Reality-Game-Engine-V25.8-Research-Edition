#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare hit-judge experiment profiles on recorded pseudo-hit evidence.

This is a lightweight regression tool. It does not replay raw perception or
reconstruct missing pseudo hits; it audits the evidence already emitted by a
live run or dataset folder and estimates how each configured profile would
count/suppress those candidates.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as fp:
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


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def ts_from_row(row: Dict[str, Any]) -> float:
    for key in ("wall_time", "timestamp_s"):
        if key in row:
            return safe_float(row.get(key), 0.0)
    for key in ("wall_us", "updated_wall_us", "dataset_ingest_wall_us", "timestamp_us"):
        if key in row:
            return safe_float(row.get(key), 0.0) / 1_000_000.0
    payload = row.get("payload")
    if isinstance(payload, dict) and payload.get("ts_ms") is not None:
        return safe_float(payload.get("ts_ms"), 0.0) / 1000.0
    return 0.0


def point_in_expanded_bbox(point: Any, bbox: Any, expand_px: float) -> bool:
    try:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return False
        x, y = float(point[0]), float(point[1])
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1 - expand_px) <= x <= (x2 + expand_px) and (y1 - expand_px) <= y <= (y2 + expand_px)
    except Exception:
        return False


def dataset_window(source: Path) -> Tuple[float, float]:
    summary = read_json(source / "summary.json")
    manifest = read_json(source / "manifest.json")
    start_s = safe_float(summary.get("started_wall_us") or manifest.get("created_wall_us"), 0.0) / 1_000_000.0
    stop_s = safe_float(summary.get("stopped_wall_us") or manifest.get("stopped_wall_us"), 0.0) / 1_000_000.0
    return start_s, stop_s


def source_files(source: Path, project_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
    root = source.resolve()
    roots = [root]
    if (root / "status").is_dir():
        roots.insert(0, root / "status")
    out: Dict[str, List[Path]] = {"pseudo_hit": [], "hit_candidate": [], "annotation": [], "reject": []}
    seen = set()
    for base in roots:
        for key, patterns in {
            "pseudo_hit": ["pseudo_hit_all.jsonl", "pseudo_hit_*.jsonl"],
            "hit_candidate": ["hit_candidate_all.jsonl", "hit_candidate_*.jsonl"],
            "annotation": ["operator_annotations.jsonl"],
            "reject": ["hit_reject_debug_*.jsonl", "hit_reject_debug_all.jsonl"],
        }.items():
            for pattern in patterns:
                for path in sorted(base.glob(pattern)):
                    if path in seen or not path.is_file():
                        continue
                    seen.add(path)
                    out[key].append(path)
    if project_dir is not None and not out["pseudo_hit"]:
        sync = project_dir.resolve() / "sync_ipc"
        for key, names in {
            "pseudo_hit": ["pseudo_hit_all.jsonl"],
            "hit_candidate": ["hit_candidate_all.jsonl", "hit_candidate_shadow_strong_reasoning_all.jsonl"],
            "reject": ["hit_reject_debug_all.jsonl"],
        }.items():
            for name in names:
                path = sync / name
                if path.is_file() and path not in seen:
                    seen.add(path)
                    out[key].append(path)
    return out


def load_rows(paths: Iterable[Path], start_s: float = 0.0, stop_s: float = 0.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for obj in iter_jsonl(path):
            if start_s > 0.0 and stop_s > start_s:
                t = ts_from_row(obj)
                if t <= 0.0 or t < start_s or t > stop_s:
                    continue
            item = dict(obj)
            item["_source_file"] = str(path)
            rows.append(item)
    return rows


def row_track_key(row: Dict[str, Any]) -> str:
    for key in ("global_bullet_id", "track_key", "stable_track_id", "track_id", "bullet_id", "global_track_id"):
        val = row.get(key)
        text = str(val).strip() if val not in (None, "") else ""
        if text and text not in {"-1", "0", "None", "none"}:
            return text
    cam = str(row.get("camera", row.get("cam", "")))
    ts = str(row.get("event_time_s", row.get("wall_time_s", row.get("timestamp_us", ""))))
    return f"{cam}:{ts}:{row.get('person_id', '')}"


def positive_contact_evidence(row: Dict[str, Any]) -> bool:
    reason = str(row.get("pseudo_hit_reason", "") or "")
    method = str(row.get("mask_method", "") or "")
    event_type = str(row.get("event_type", "") or "")
    point_count = safe_int(row.get("track_point_count"), 0)
    path_len = safe_float(row.get("track_path_len_px"), 0.0)
    terminal_or_turn = event_type in {"terminate", "turn", "EVENT_TERMINATE", "EVENT_TURN"}
    stable_visual = reason in {
        "visual_point_stable_track_inside_mask",
        "visual_point_stable_segment_cross_mask",
    }
    keypoint_inside = method in {
        "point_in_mask",
        "segment_intersect_mask",
        "track_point_inside_mask",
        "track_polyline_cross_mask",
    }
    return bool((terminal_or_turn or stable_visual) and keypoint_inside and (point_count >= 3 or path_len >= 20.0))


def recompute_track_start_inside(row: Dict[str, Any], bbox_px: float) -> bool:
    point = row.get("human_noise_track_start_point")
    bbox = row.get("person_bbox") or row.get("bbox_xyxy") or row.get("bbox")
    computed = point_in_expanded_bbox(point, bbox, bbox_px)
    if computed:
        return True
    if bbox_px == safe_float(row.get("human_noise_track_start_bbox_expand_px"), bbox_px):
        return safe_bool(row.get("human_noise_track_start_inside_expanded_bbox"), False)
    return False


def adjusted_noise_score(row: Dict[str, Any], inside: bool, policy: str) -> float:
    score = safe_float(row.get("human_noise_score"), 0.0)
    reasons = row.get("human_noise_reasons", [])
    had_legacy_start = safe_bool(row.get("human_noise_track_start_inside_expanded_bbox"), False)
    if isinstance(reasons, list):
        had_legacy_start = had_legacy_start or ("track_start_inside_expanded_bbox" in {str(x) for x in reasons})
    elif isinstance(reasons, str):
        had_legacy_start = had_legacy_start or ("track_start_inside_expanded_bbox" in reasons)
    if had_legacy_start:
        score -= 0.50
    if inside and policy != "off":
        score += 0.50 if policy == "hard" else 0.28
    return max(0.0, min(1.0, score))


def profile_promotes(row: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[bool, str]:
    mode = str(profile.get("human_noise_filter_mode", "soft") or "soft")
    start_filter = safe_bool(profile.get("human_noise_track_start_bbox_filter_enable"), True)
    bbox_px = safe_float(profile.get("human_noise_track_start_bbox_expand_px"), 24.0)
    authority = str(profile.get("final_hit_authority", "strong_gate") or "strong_gate")
    policy = str(profile.get("human_noise_track_start_bbox_policy", "soft") or "soft")
    if policy not in {"soft", "hard", "off"}:
        policy = "soft"
    if authority != "strong_reasoning":
        return False, "authority_not_strong_reasoning"
    if mode in {"off", "shadow"}:
        return True, f"mode_{mode}"
    inside = recompute_track_start_inside(row, bbox_px)
    score = adjusted_noise_score(row, inside, policy)
    clear_noise = safe_bool(row.get("human_noise_clear"), False)
    inbound = safe_bool(row.get("inbound_evidence"), False)
    suspect = safe_bool(row.get("human_noise_suspect"), False) or inside
    require_inbound = safe_bool(profile.get("human_noise_require_inbound_for_low_quality"), True)
    positive = positive_contact_evidence(row)
    if start_filter and inside and policy == "hard":
        return False, "track_start_inside_expanded_bbox"
    if mode == "hard":
        if score >= safe_float(profile.get("human_noise_soft_threshold"), 0.62):
            return False, "hard_score_threshold"
        if require_inbound and (not inbound) and suspect:
            return False, "hard_require_inbound"
    else:
        if score >= safe_float(profile.get("human_noise_hard_threshold"), 0.88) and not positive:
            return False, "soft_high_score_threshold"
        if require_inbound and clear_noise and not positive:
            return False, "soft_clear_noise_without_inbound"
    return True, "promoted"


def match_annotations(annotations: List[Dict[str, Any]], events: List[Dict[str, Any]], kind: str, window_s: float) -> int:
    ann = [r for r in annotations if str(r.get("kind", r.get("type", ""))) == kind]
    pairs = []
    for i, a in enumerate(ann):
        at = ts_from_row(a)
        for j, ev in enumerate(events):
            dt = ts_from_row(ev) - at
            if abs(dt) <= window_s:
                pairs.append((abs(dt), i, j))
    used_a, used_e = set(), set()
    for _, i, j in sorted(pairs):
        if i in used_a or j in used_e:
            continue
        used_a.add(i)
        used_e.add(j)
    return len(used_a)


def annotations_with_near_event(annotations: List[Dict[str, Any]], events: List[Dict[str, Any]], kind: str, window_s: float) -> int:
    count = 0
    for ann in annotations:
        if str(ann.get("kind", ann.get("type", ""))) != kind:
            continue
        at = ts_from_row(ann)
        if any(abs(ts_from_row(ev) - at) <= window_s for ev in events):
            count += 1
    return count


def profile_result(rows: List[Dict[str, Any]], annotations: List[Dict[str, Any]], profile: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(profile.get("human_noise_filter_mode", "soft") or "soft")
    bbox_px = safe_float(profile.get("human_noise_track_start_bbox_expand_px"), 24.0)
    authority = str(profile.get("final_hit_authority", "strong_gate") or "strong_gate")
    policy = str(profile.get("human_noise_track_start_bbox_policy", "soft") or "soft")
    suppress_count = 0
    promoted_count = 0
    suspect_count = 0
    reason_counter: Counter[str] = Counter()
    suppress_reasons: Counter[str] = Counter()
    track_ids = set()
    per_camera: Counter[str] = Counter()
    promoted_rows: List[Dict[str, Any]] = []
    for row in rows:
        track_ids.add(row_track_key(row))
        cam = str(row.get("camera", row.get("cam", "")) or "")
        if cam:
            per_camera[cam] += 1
        reasons = row.get("human_noise_reasons", row.get("final_hit_human_noise_reason", []))
        if isinstance(reasons, str):
            reason_counter[reasons] += 1
        elif isinstance(reasons, list):
            for r in reasons:
                reason_counter[str(r)] += 1
        inside = recompute_track_start_inside(row, bbox_px)
        suspect = safe_bool(row.get("human_noise_suspect"), False) or inside
        if suspect:
            suspect_count += 1
        promoted, why = profile_promotes(row, profile)
        if not promoted:
            suppress_count += 1
            suppress_reasons[why] += 1
        if promoted:
            promoted_count += 1
            promoted_rows.append(row)
    return {
        "profile_id": profile.get("experiment_id", profile.get("name", "inline")),
        "stage": profile.get("stage", ""),
        "authority": authority,
        "human_noise_mode": mode,
        "bbox_start_expand_px": bbox_px,
        "bbox_start_policy": policy,
        "pseudo_hit_rows": len(rows),
        "unique_track_keys": len(track_ids),
        "human_noise_suspect": suspect_count,
        "estimated_suppressed": suppress_count,
        "estimated_promoted_final": promoted_count,
        "hit_annotations_matched_2s": match_annotations(annotations, promoted_rows, "hit", 2.0),
        "miss_annotations_with_promoted_2s": annotations_with_near_event(annotations, promoted_rows, "miss", 2.0),
        "suppress_reasons": dict(suppress_reasons.most_common(20)),
        "per_camera": dict(sorted(per_camera.items())),
        "human_noise_reasons": dict(reason_counter.most_common(20)),
        "note": "Counts are based on recorded pseudo-hit evidence only; this does not recover candidates that were never emitted.",
    }


def source_summary(source: Path, profiles: List[Dict[str, Any]], project_dir: Optional[Path] = None) -> Dict[str, Any]:
    files = source_files(source, project_dir)
    start_s, stop_s = dataset_window(source)
    pseudo = load_rows(files["pseudo_hit"], start_s, stop_s)
    candidates = load_rows(files["hit_candidate"], start_s, stop_s)
    annotations = load_rows(files["annotation"], start_s, stop_s)
    rejects = load_rows(files["reject"], start_s, stop_s)
    annotation_counts = Counter(str(x.get("kind", x.get("type", ""))) for x in annotations)
    base = {
        "source_dir": str(source.resolve()),
        "files": {k: [str(p) for p in v] for k, v in files.items()},
        "counts": {
            "pseudo_hit_rows": len(pseudo),
            "hit_candidate_rows": len(candidates),
            "annotation_rows": len(annotations),
            "reject_rows": len(rejects),
            "annotation_by_kind": dict(annotation_counts),
        },
        "pseudo_hit_reason_total": dict(Counter(str(x.get("pseudo_hit_reason", "")) for x in pseudo).most_common(30)),
        "final_hit_authority_total": dict(Counter(str(x.get("final_hit_authority", "")) for x in pseudo).most_common(10)),
        "profile_results": [profile_result(pseudo, annotations, p) for p in profiles],
    }
    return base


def render_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# Hit Judge Experiment Compare",
        "",
        "| source | profile | authority | mode | policy | bbox_px | pseudo | suppressed | promoted | hit@2s | miss_near@2s | tracks |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src in payload.get("sources", []):
        name = Path(str(src.get("source_dir", ""))).name
        for row in src.get("profile_results", []):
            lines.append(
                f"| {name} | {row.get('profile_id')} | {row.get('authority')} | {row.get('human_noise_mode')} | "
                f"{row.get('bbox_start_policy')} | {row.get('bbox_start_expand_px')} | {row.get('pseudo_hit_rows')} | "
                f"{row.get('estimated_suppressed')} | {row.get('estimated_promoted_final')} | "
                f"{row.get('hit_annotations_matched_2s')} | {row.get('miss_annotations_with_promoted_2s')} | {row.get('unique_track_keys')} |"
            )
    lines.append("")
    lines.append("Notes:")
    lines.append("- This report audits emitted pseudo-hit evidence; it is a fast regression aid, not a raw sensor replay.")
    lines.append("- Use it before remote live tests to catch obvious parameter regressions.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare hit-judge experiment profiles on recorded pseudo-hit evidence.")
    ap.add_argument("--source-dir", action="append", default=[], help="Run or dataset folder containing pseudo_hit*.jsonl.")
    ap.add_argument("--source-root", default="", help="Optional root containing run/dataset folders.")
    ap.add_argument("--project-dir", default="", help="Project root; used to read sync_ipc logs when source-dir is a dataset folder.")
    ap.add_argument("--name-contains", action="append", default=[])
    ap.add_argument("--profile", action="append", default=[], help="Hit-judge experiment profile JSON.")
    ap.add_argument("--output-root", default="")
    args = ap.parse_args()

    profiles = [read_json(Path(p).expanduser()) for p in args.profile]
    if not profiles:
        profiles = [
            {"experiment_id": "inline_bbox24_soft_policy", "final_hit_authority": "strong_reasoning", "human_noise_filter_mode": "soft", "human_noise_track_start_bbox_filter_enable": True, "human_noise_track_start_bbox_expand_px": 24.0, "human_noise_track_start_bbox_policy": "soft"},
            {"experiment_id": "inline_bbox48_soft_policy", "final_hit_authority": "strong_reasoning", "human_noise_filter_mode": "soft", "human_noise_track_start_bbox_filter_enable": True, "human_noise_track_start_bbox_expand_px": 48.0, "human_noise_track_start_bbox_policy": "soft"},
            {"experiment_id": "inline_bbox96_soft_policy", "final_hit_authority": "strong_reasoning", "human_noise_filter_mode": "soft", "human_noise_track_start_bbox_filter_enable": True, "human_noise_track_start_bbox_expand_px": 96.0, "human_noise_track_start_bbox_policy": "soft"},
            {"experiment_id": "inline_bbox24_hard_policy", "final_hit_authority": "strong_reasoning", "human_noise_filter_mode": "soft", "human_noise_track_start_bbox_filter_enable": True, "human_noise_track_start_bbox_expand_px": 24.0, "human_noise_track_start_bbox_policy": "hard"},
            {"experiment_id": "inline_bbox48_hard_policy", "final_hit_authority": "strong_reasoning", "human_noise_filter_mode": "soft", "human_noise_track_start_bbox_filter_enable": True, "human_noise_track_start_bbox_expand_px": 48.0, "human_noise_track_start_bbox_policy": "hard"},
            {"experiment_id": "inline_bbox96_hard_policy", "final_hit_authority": "strong_reasoning", "human_noise_filter_mode": "soft", "human_noise_track_start_bbox_filter_enable": True, "human_noise_track_start_bbox_expand_px": 96.0, "human_noise_track_start_bbox_policy": "hard"},
        ]
    sources = [Path(p).expanduser().resolve() for p in args.source_dir]
    if args.source_root:
        root = Path(args.source_root).expanduser().resolve()
        filters = [str(x).lower() for x in args.name_contains if str(x).strip()]
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if filters and not any(f in child.name.lower() for f in filters):
                continue
            if any(child.glob("pseudo_hit*.jsonl")) or (child / "status").is_dir():
                sources.append(child)
    if not sources:
        raise SystemExit("no source dirs provided")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser() if args.output_root else (sources[0].parent / f"_hit_compare_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else None
    payload = {
        "schema_version": 1,
        "kind": "hit_judge_experiment_compare",
        "created_wall_us": int(time.time() * 1_000_000),
        "output_root": str(output_root.resolve()),
        "profiles": profiles,
        "sources": [source_summary(s, profiles, project_dir) for s in sources],
    }
    (output_root / "hit_experiment_compare.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "hit_experiment_compare.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "output_root": str(output_root.resolve()),
        "source_count": len(sources),
        "report": str((output_root / "hit_experiment_compare.md").resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
