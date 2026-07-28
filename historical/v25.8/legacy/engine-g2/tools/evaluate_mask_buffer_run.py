#!/usr/bin/env python3
"""Summarize event human-mask buffer telemetry for one evaluation run."""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import math
import re
import zipfile
from pathlib import Path


STATS_RE = re.compile(r"event_human_mask_stats_([^/\\]+)\.jsonl$", re.IGNORECASE)
STATUS_RE = re.compile(r"event_worker_([^/\\]+)_status\.json$", re.IGNORECASE)


def _pct(values, q):
    values = sorted(float(v) for v in values if _is_num(v))
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    pos = (len(values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(values[lo], 3)
    frac = pos - lo
    return round(values[lo] * (1.0 - frac) + values[hi] * frac, 3)


def _is_num(v):
    try:
        return v is not None and math.isfinite(float(v))
    except Exception:
        return False


def _load_json_line(raw):
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _iter_files(inputs):
    for root in inputs:
        p = Path(root)
        if p.is_file():
            yield p
        elif p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    yield child


def _iter_records_from_path(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                m_stats = STATS_RE.search(name)
                m_status = STATUS_RE.search(name)
                if not (m_stats or m_status):
                    continue
                with zf.open(name, "r") as fh:
                    if m_stats:
                        for raw_b in fh:
                            rec = _load_json_line(raw_b.decode("utf-8", errors="replace"))
                            if rec is not None:
                                yield "stats", m_stats.group(1), rec, f"{path}!{name}"
                    else:
                        rec = _load_json_line(fh.read().decode("utf-8", errors="replace"))
                        if rec is not None:
                            yield "status", m_status.group(1), rec, f"{path}!{name}"
        return

    name = str(path)
    m_stats = STATS_RE.search(name)
    m_status = STATUS_RE.search(name)
    if m_stats:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                rec = _load_json_line(raw)
                if rec is not None:
                    yield "stats", m_stats.group(1), rec, str(path)
    elif m_status:
        rec = _load_json_line(path.read_text(encoding="utf-8", errors="replace"))
        if rec is not None:
            yield "status", m_status.group(1), rec, str(path)


def _reason_count(records, key):
    c = collections.Counter()
    for rec in records:
        v = str(rec.get(key, "") or "")
        if v:
            c[v] += 1
    return dict(c.most_common())


def _reason_family(reason):
    text = str(reason or "")
    if not text:
        return ""
    return text.split(":", 1)[0]


def _summarize_camera(records):
    n = len(records)
    if n <= 0:
        return {"n": 0}
    n_in = [int(rec.get("n_in", 0) or 0) for rec in records]
    n_out = [int(rec.get("n_out", 0) or 0) for rec in records]
    n_drop = [int(rec.get("n_drop", 0) or 0) for rec in records]
    drop_pct = [float(rec.get("drop_pct", 0.0) or 0.0) for rec in records]
    ages = [rec.get("buffer_slice_age_ms") for rec in records if _is_num(rec.get("buffer_slice_age_ms"))]
    buffer_records = [rec for rec in records if rec.get("buffer_enabled") is True]
    release_counts = collections.Counter(str(rec.get("buffer_release_reason", "") or "") for rec in buffer_records)
    release_counts.pop("", None)
    same_sync = sum(v for k, v in release_counts.items() if k.startswith("same_sync_ready"))
    raw_release = sum(
        v for k, v in release_counts.items()
        if k in ("budget_raw", "timeout_raw", "max_events_force_raw", "max_buckets_force_raw")
    )
    forced = sum(1 for rec in buffer_records if rec.get("buffer_forced_release") is True)
    missing = sum(1 for rec in records if str(rec.get("reason", "") or "").startswith("current_mask_missing"))
    before_frame = sum(1 for rec in records if str(rec.get("reason", "") or "").startswith("before_frame_window"))
    empty_current = sum(1 for rec in records if str(rec.get("reason", "") or "").startswith("empty_current_mask"))
    applied = sum(1 for rec in records if rec.get("applied") is True)
    reason_families = collections.Counter(_reason_family(rec.get("reason", "")) for rec in records)
    reason_families.pop("", None)
    total_in = sum(n_in)
    total_out = sum(n_out)
    total_drop = sum(n_drop)
    return {
        "n": n,
        "total_in": int(total_in),
        "total_out": int(total_out),
        "total_drop": int(total_drop),
        "event_drop_ratio": round(float(total_drop) / max(1.0, float(total_in)), 6),
        "mask_applied_ratio": round(applied / max(1, n), 6),
        "current_mask_missing_ratio": round(missing / max(1, n), 6),
        "before_frame_window_ratio": round(before_frame / max(1, n), 6),
        "empty_current_mask_ratio": round(empty_current / max(1, n), 6),
        "drop_pct_p50": _pct(drop_pct, 0.50),
        "drop_pct_p95": _pct(drop_pct, 0.95),
        "buffer_records": len(buffer_records),
        "buffer_same_sync_ratio": round(same_sync / max(1, len(buffer_records)), 6),
        "buffer_raw_release_ratio": round(raw_release / max(1, len(buffer_records)), 6),
        "buffer_forced_release_ratio": round(forced / max(1, len(buffer_records)), 6),
        "buffer_age_ms_p50": _pct(ages, 0.50),
        "buffer_age_ms_p95": _pct(ages, 0.95),
        "reason_counts": _reason_count(records, "reason"),
        "reason_family_counts": dict(reason_families.most_common()),
        "buffer_release_counts": dict(release_counts.most_common()),
    }


def summarize(inputs):
    records_by_camera_source = collections.defaultdict(lambda: collections.defaultdict(list))
    status_by_camera = {}
    sources = collections.Counter()
    for path in _iter_files(inputs):
        for kind, camera, rec, source in _iter_records_from_path(path):
            sources[source] += 1
            if kind == "stats":
                records_by_camera_source[str(camera)][str(source)].append(rec)
            else:
                status_by_camera[str(camera)] = rec

    records_by_camera = {}
    selected_sources = {}
    for cam, by_source in records_by_camera_source.items():
        plain_sources = [src for src in by_source if "!" not in src]
        chosen = plain_sources if plain_sources else list(by_source)
        records = []
        for src in sorted(chosen):
            records.extend(by_source[src])
        records_by_camera[cam] = records
        selected_sources[cam] = sorted(chosen)

    cameras = sorted(set(records_by_camera) | set(status_by_camera))
    per_camera = {cam: _summarize_camera(records_by_camera.get(cam, [])) for cam in cameras}
    all_records = []
    for recs in records_by_camera.values():
        all_records.extend(recs)
    overall = _summarize_camera(all_records)
    latest_status = {
        cam: {
            "state": status_by_camera[cam].get("state"),
            "active_reason": status_by_camera[cam].get("active_reason"),
            "human_mask_buffer": status_by_camera[cam].get("human_mask_buffer", {}),
            "human_mask": status_by_camera[cam].get("human_mask", {}),
        }
        for cam in sorted(status_by_camera)
    }
    return {
        "inputs": [str(Path(p)) for p in inputs],
        "discovered_source_count": len(sources),
        "source_count": sum(len(srcs) for srcs in selected_sources.values()),
        "sources": dict(sources),
        "selected_stats_sources": selected_sources,
        "overall": overall,
        "per_camera": per_camera,
        "latest_status": latest_status,
    }


def write_markdown(summary, out_md, run_name):
    overall = summary["overall"]
    lines = [
        f"# Event Human-Mask Buffer Evaluation: {run_name}",
        "",
        f"- generated_at: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"- selected_source_count: {summary['source_count']}",
        f"- discovered_source_count: {summary.get('discovered_source_count', summary['source_count'])}",
        "",
        "## Overall",
        "",
        f"- slices: {overall.get('n', 0)}",
        f"- mask_applied_ratio: {overall.get('mask_applied_ratio')}",
        f"- current_mask_missing_ratio: {overall.get('current_mask_missing_ratio')}",
        f"- before_frame_window_ratio: {overall.get('before_frame_window_ratio')}",
        f"- empty_current_mask_ratio: {overall.get('empty_current_mask_ratio')}",
        f"- event_drop_ratio: {overall.get('event_drop_ratio')}",
        f"- buffer_same_sync_ratio: {overall.get('buffer_same_sync_ratio')}",
        f"- buffer_raw_release_ratio: {overall.get('buffer_raw_release_ratio')}",
        f"- buffer_forced_release_ratio: {overall.get('buffer_forced_release_ratio')}",
        f"- buffer_age_ms_p50/p95: {overall.get('buffer_age_ms_p50')} / {overall.get('buffer_age_ms_p95')}",
        "",
        "## Per Camera",
        "",
        "| camera | slices | applied | missing | before-frame | empty-mask | same-sync | raw-release | forced | age p50 | age p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cam, item in summary["per_camera"].items():
        lines.append(
            f"| {cam} | {item.get('n', 0)} | {item.get('mask_applied_ratio')} | "
            f"{item.get('current_mask_missing_ratio')} | {item.get('before_frame_window_ratio')} | "
            f"{item.get('empty_current_mask_ratio')} | {item.get('buffer_same_sync_ratio')} | "
            f"{item.get('buffer_raw_release_ratio')} | {item.get('buffer_forced_release_ratio')} | "
            f"{item.get('buffer_age_ms_p50')} | {item.get('buffer_age_ms_p95')} |"
        )
    lines += ["", "## Release Reasons", ""]
    for cam, item in summary["per_camera"].items():
        lines.append(f"- {cam}: {json.dumps(item.get('buffer_release_counts', {}), ensure_ascii=False)}")
    lines += ["", "## Reason Families", ""]
    for cam, item in summary["per_camera"].items():
        lines.append(f"- {cam}: {json.dumps(item.get('reason_family_counts', {}), ensure_ascii=False)}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Directories or zip files containing event_human_mask_stats/status files.")
    ap.add_argument("--run-name", default="", help="Name for output files.")
    ap.add_argument("--out-dir", default="_sandbox/event_mask_buffer_90ms_feasibility/eval_outputs")
    args = ap.parse_args()

    run_name = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = summarize(args.inputs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{run_name}_metrics.json"
    out_md = out_dir / f"{run_name}_report.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_md, run_name)
    compact_overall = {
        k: v for k, v in summary.get("overall", {}).items()
        if k not in ("reason_counts", "reason_family_counts", "buffer_release_counts")
    }
    print(json.dumps({
        "run_name": run_name,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "selected_source_count": summary.get("source_count"),
        "discovered_source_count": summary.get("discovered_source_count"),
        "overall": compact_overall,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
