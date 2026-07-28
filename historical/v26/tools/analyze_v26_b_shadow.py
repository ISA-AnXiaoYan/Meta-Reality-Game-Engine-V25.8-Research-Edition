#!/usr/bin/env python3
"""Summarize V26-B polygon-gate and business-ID shadow evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            continue


def find_logs(run_dir: Path, stem: str, camera: str) -> list[Path]:
    wanted = f"{stem}_{camera}.jsonl"
    return sorted(path for path in run_dir.rglob(wanted) if path.is_file())


def summarize_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = Counter(str(row.get("source", "NONE") or "NONE") for row in rows)
    reason = Counter(str(row.get("reason", "") or "") for row in rows)
    input_count = sum(int(row.get("input_cluster_count", 0) or 0) for row in rows)
    filtered_count = sum(int(row.get("filtered_cluster_count", 0) or 0) for row in rows)
    return {
        "rows": len(rows),
        "source": dict(sorted(source.items())),
        "reason_top": reason.most_common(8),
        "input_cluster_count": input_count,
        "filtered_cluster_count": filtered_count,
        "filtered_ratio": round(filtered_count / max(1, input_count), 6),
        "max_hold_age_us": max((int(row.get("hold_age_us", 0) or 0) for row in rows), default=0),
        "max_hold_sync_gap": max((int(row.get("hold_sync_gap", 0) or 0) for row in rows), default=0),
    }


def summarize_v1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    max_active = max_confirmed = max_cloud = collision_links = 0
    for row in rows:
        max_active = max(max_active, int(row.get("active_ids", 0) or 0))
        max_confirmed = max(max_confirmed, int(row.get("confirmed_ids", 0) or 0))
        max_cloud = max(max_cloud, int(row.get("cloud_suspect_ids", 0) or 0))
        collision_links += int(row.get("collision_links", 0) or 0)
        for item in row.get("assignments", []) or []:
            if isinstance(item, dict):
                totals[str(item.get("status", "unknown"))] += 1
                if bool(item.get("cloud_suspect", False)):
                    totals["cloud_suspect_assignment"] += 1
    return {
        "rows": len(rows),
        "assignment_status": dict(sorted(totals.items())),
        "max_active_ids": max_active,
        "max_confirmed_ids": max_confirmed,
        "max_cloud_suspect_ids": max_cloud,
        "collision_links": collision_links,
    }


def summarize_v3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter()
    size_jump = 0
    for row in rows:
        for item in row.get("assignments", []) or []:
            if not isinstance(item, dict):
                continue
            status[str(item.get("status", "unknown"))] += 1
            if bool(item.get("size_jump", False)):
                size_jump += 1
    return {"rows": len(rows), "assignment_status": dict(sorted(status.items())), "size_jump_count": size_jump}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Fetched remote run archive directory or sync_ipc root")
    parser.add_argument("--camera", default="E")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.run_dir).expanduser().resolve()
    camera = str(args.camera).strip().upper()
    gate_paths = find_logs(root, "v26_b_polygon_gate", camera)
    v1_paths = find_logs(root, "v26_b_business_id_v1", camera)
    v3_paths = find_logs(root, "v26_b_business_id_v3", camera)
    report = {
        "schema_version": 1,
        "patch": "V26-B",
        "camera": camera,
        "run_dir": str(root),
        "files": {
            "polygon_gate": [str(path) for path in gate_paths],
            "business_id_v1": [str(path) for path in v1_paths],
            "business_id_v3": [str(path) for path in v3_paths],
        },
        "polygon_gate": summarize_gate(list(iter_jsonl(gate_paths))),
        "business_id_v1_shadow": summarize_v1(list(iter_jsonl(v1_paths))),
        "business_id_v3_shadow": summarize_v3(list(iter_jsonl(v3_paths))),
    }
    report["instrumentation_pass"] = bool(
        report["polygon_gate"]["rows"] > 0
        and report["business_id_v1_shadow"]["rows"] > 0
        and report["business_id_v3_shadow"]["rows"] > 0
    )
    observed_assignments = sum(report["business_id_v1_shadow"]["assignment_status"].values())
    report["effectiveness_observed"] = bool(
        report["polygon_gate"]["filtered_cluster_count"] > 0 or observed_assignments > 0
    )
    report["report_status"] = (
        "effective_shadow_observed"
        if report["instrumentation_pass"] and report["effectiveness_observed"]
        else ("infrastructure_pass_no_target" if report["instrumentation_pass"] else "missing_shadow_evidence")
    )
    report["pass"] = bool(report["instrumentation_pass"])
    output = Path(args.output).expanduser() if args.output else root / f"v26_b_shadow_report_{camera}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
