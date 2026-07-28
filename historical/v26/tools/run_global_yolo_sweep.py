#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run short Global YOLO C+3 parameter sweeps and summarize FPS.

The script creates a temporary launch profile per case, runs the normal launcher,
then copies Global YOLO status/perf artifacts into sync_ipc/yolo_sweeps.
It intentionally disables automatic CPU hotspot collection so the sweep itself
does not add a heavy extra process to every case.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_csv_floats(value: str) -> List[float]:
    out: List[float] = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_csv_ints(value: str) -> List[int]:
    return [int(round(x)) for x in _parse_csv_floats(value)]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as fp:
            obj = json.load(fp)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst_dir / src.name)
    except Exception:
        pass


def _percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(100.0, pct)) / 100.0 * (len(vals) - 1)
    lo = int(pos)
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _load_perf_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            for row in csv.DictReader(fp):
                if not row:
                    continue
                rows.append(dict(row))
    except Exception:
        return rows
    return rows


def _summarize_perf(perf_paths: Sequence[Path], warmup_sec: float = 20.0) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    by_worker: List[Dict[str, Any]] = []
    for idx, path in enumerate(perf_paths):
        rows = _load_perf_rows(path)
        if not rows:
            by_worker.append({"worker": idx, "rows": 0, "frames": 0})
            continue
        wall = [_safe_int(r.get("wall_us")) for r in rows if _safe_int(r.get("wall_us")) > 0]
        min_wall = min(wall) if wall else 0
        filtered = [
            r for r in rows
            if _safe_int(r.get("wall_us")) <= 0 or _safe_int(r.get("wall_us")) - min_wall >= int(warmup_sec * 1_000_000)
        ]
        if not filtered and rows:
            filtered = rows
        all_rows.extend(filtered)
        frames = sum(_safe_int(r.get("batch_size")) for r in filtered)
        full = sum(1 for r in filtered if _safe_int(r.get("batch_size")) >= _safe_int(r.get("group_size"), _safe_int(r.get("batch_size"))))
        partial = len(filtered) - full
        worker_wall = [_safe_int(r.get("wall_us")) for r in filtered if _safe_int(r.get("wall_us")) > 0]
        worker_elapsed = (max(worker_wall) - min(worker_wall)) / 1_000_000.0 if len(worker_wall) >= 2 else 0.0
        by_worker.append({
            "worker": idx,
            "path": str(path),
            "rows": len(filtered),
            "frames": frames,
            "elapsed_sec": round(worker_elapsed, 3),
            "fps": round(frames / worker_elapsed, 3) if worker_elapsed > 0 else 0.0,
            "full_batches": full,
            "partial_batches": partial,
            "full_batch_rate": round(full / max(1, len(filtered)), 4),
        })
    if not all_rows:
        return {"rows": 0, "workers": by_worker}

    wall = [_safe_int(r.get("wall_us")) for r in all_rows if _safe_int(r.get("wall_us")) > 0]
    elapsed = (max(wall) - min(wall)) / 1_000_000.0 if len(wall) >= 2 else 0.0
    frames = sum(_safe_int(r.get("batch_size")) for r in all_rows)
    full = sum(1 for r in all_rows if _safe_int(r.get("batch_size")) >= _safe_int(r.get("group_size"), _safe_int(r.get("batch_size"))))
    partial = len(all_rows) - full
    batch_sizes = [_safe_int(r.get("batch_size")) for r in all_rows]
    group_sizes = [_safe_int(r.get("group_size")) for r in all_rows]
    collect_wait = [_safe_float(r.get("collect_wait_ms")) for r in all_rows]
    predict_ms = [_safe_float(r.get("predict_ms")) for r in all_rows if _safe_float(r.get("predict_ms")) > 0]
    post_ms = [_safe_float(r.get("postprocess_ms")) for r in all_rows if _safe_float(r.get("postprocess_ms")) >= 0]
    total_ms = [_safe_float(r.get("total_ms")) for r in all_rows if _safe_float(r.get("total_ms")) > 0]
    return {
        "rows": len(all_rows),
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "fps_total": round(frames / elapsed, 3) if elapsed > 0 else 0.0,
        "full_batches": full,
        "partial_batches": partial,
        "full_batch_rate": round(full / max(1, len(all_rows)), 4),
        "batch_size_avg": round(statistics.fmean(batch_sizes), 3) if batch_sizes else 0.0,
        "group_size_avg": round(statistics.fmean(group_sizes), 3) if group_sizes else 0.0,
        "collect_wait_ms_avg": round(statistics.fmean(collect_wait), 3) if collect_wait else 0.0,
        "collect_wait_ms_p95": round(_percentile(collect_wait, 95), 3),
        "predict_ms_avg": round(statistics.fmean(predict_ms), 3) if predict_ms else 0.0,
        "predict_ms_p95": round(_percentile(predict_ms, 95), 3),
        "postprocess_ms_avg": round(statistics.fmean(post_ms), 3) if post_ms else 0.0,
        "total_ms_avg": round(statistics.fmean(total_ms), 3) if total_ms else 0.0,
        "total_ms_p95": round(_percentile(total_ms, 95), 3),
        "workers": by_worker,
    }


def _status_summary(status_path: Path) -> Dict[str, Any]:
    status = _read_json(status_path)
    stats = status.get("stats", {}) if isinstance(status.get("stats"), dict) else {}
    workers = status.get("workers", []) if isinstance(status.get("workers"), list) else []
    return {
        "state": status.get("state"),
        "global_yolo_primary": status.get("global_yolo_primary"),
        "merge_publisher": status.get("merge_publisher"),
        "legacy_yolo_skipped": status.get("legacy_yolo_skipped"),
        "postprocess_backends": status.get("postprocess_backends"),
        "postprocess_draw_enabled": status.get("postprocess_draw_enabled"),
        "cpu_draw_postprocess": status.get("cpu_draw_postprocess"),
        "mask_postprocess_enabled": status.get("mask_postprocess_enabled"),
        "tensor_rt_enabled": status.get("tensor_rt_enabled"),
        "worker_count": status.get("worker_count"),
        "worker_states": status.get("worker_states"),
        "frames_inferred": stats.get("frames_inferred"),
        "records_written": stats.get("records_written"),
        "merged_records": stats.get("merged_records"),
        "workers": [
            {
                "worker": w.get("worker"),
                "state": w.get("state"),
                "stale": w.get("stale"),
                "cams": w.get("cams"),
                "stats": w.get("stats"),
            }
            for w in workers
            if isinstance(w, dict)
        ],
    }


def _make_case_profile(base: Dict[str, Any], wait_ms: float, target_fps: int, out_path: Path, args: argparse.Namespace) -> None:
    obj = json.loads(json.dumps(base))
    cfg = obj.setdefault("args", {})
    cfg["global_yolo_batch_collect_wait_ms"] = float(wait_ms)
    cfg["global_yolo_target_fps"] = int(target_fps)
    cfg["auto_cpu_hotspots_enable"] = False
    cfg["event_startup_diagnostics_window_enable"] = False
    cfg["global_yolo_enable"] = True
    cfg["global_yolo_shadow_mode"] = False
    cfg["global_yolo_primary_workers"] = 2
    cfg["global_yolo_primary_groups"] = "A,C,E,G:B,D,F,H"
    cfg["global_yolo_merge_enable"] = True
    cfg["global_yolo_retina_masks"] = False
    cfg["global_yolo_include_mask_polygons"] = False
    cfg["global_yolo_engine"] = ""
    if args.disable_bev:
        cfg["global_bev_enable"] = False
    if args.disable_preview:
        cfg["preview_backend"] = "python-gl"
        cfg["skip_gl"] = True
    _write_json(out_path, obj)


def _terminate_tree(proc: subprocess.Popen[Any], grace_sec: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        proc.terminate()
    except Exception:
        pass
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        proc.kill()
    except Exception:
        pass


def _run_case(project_root: Path, profile_path: Path, case_dir: Path, duration_sec: float, display: str, extra_args: Sequence[str]) -> int:
    log_path = case_dir / "launcher_stdout.log"
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if display:
        env["DISPLAY"] = display
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    env.setdefault("__GL_SYNC_TO_VBLANK", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        "python3",
        "launch_fusion_system.py",
        "--launch-profile",
        str(profile_path),
        "--console-log-mode",
        "startup",
        *extra_args,
    ]
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("[sweep] cmd=" + " ".join(cmd) + "\n")
        log.write(f"[sweep] duration_sec={duration_sec} display={display}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        time.sleep(max(1.0, float(duration_sec)))
        _terminate_tree(proc)
        rc = proc.wait(timeout=10)
        log.write(f"[sweep] rc={rc}\n")
        return int(rc)


def _collect_case_artifacts(project_root: Path, case_dir: Path) -> Dict[str, Any]:
    sync = project_root / "sync_ipc"
    artifacts = case_dir / "artifacts"
    for name in [
        "global_yolo_status.json",
        "global_yolo_latest.json",
        "global_yolo_worker_0_status.json",
        "global_yolo_worker_1_status.json",
        "global_yolo_worker_0_latest.json",
        "global_yolo_worker_1_latest.json",
        "global_yolo_worker_0_perf.csv",
        "global_yolo_worker_1_perf.csv",
        "global_person_status.json",
        "global_bev_status.json",
        "trigger_status.json",
        "event_overlay_overview_status.json",
    ]:
        _copy_if_exists(sync / name, artifacts)
    log_dir = sync / "launcher_logs"
    if log_dir.exists():
        dst_logs = artifacts / "launcher_logs"
        dst_logs.mkdir(parents=True, exist_ok=True)
        for name in [
            "global_yolo_0.log",
            "global_yolo_1.log",
            "global_yolo_merge.log",
            "gl_cuda.log",
            "event_overview.log",
            "global_bev.log",
            "event.log",
            "trigger.log",
        ]:
            _copy_if_exists(log_dir / name, dst_logs)
    latest_status_dirs = sorted((sync / "status_logs").glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if latest_status_dirs:
        dst_status = artifacts / "status_logs_latest"
        dst_status.mkdir(parents=True, exist_ok=True)
        for name in ["status_latest.json", "status_summary.csv", "status_tabs.jsonl"]:
            _copy_if_exists(latest_status_dirs[0] / name, dst_status)

    perf_paths = [artifacts / "global_yolo_worker_0_perf.csv", artifacts / "global_yolo_worker_1_perf.csv"]
    summary = {
        "status": _status_summary(artifacts / "global_yolo_status.json"),
        "perf": _summarize_perf(perf_paths),
    }
    _write_json(case_dir / "summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Global YOLO C+3 wait/target FPS sweep.")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--profile", default="launch_profiles/627_event_overlay_bev.json")
    ap.add_argument("--wait-ms", default="0,0.5,1,2")
    ap.add_argument("--target-fps", default="140,160,180")
    ap.add_argument("--duration-sec", type=float, default=120.0)
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    ap.add_argument("--run-id", default="")
    ap.add_argument("--extra-launch-arg", action="append", default=[])
    ap.add_argument("--disable-bev", action="store_true", help="Disable Global BEV during sweep.")
    ap.add_argument("--disable-preview", action="store_true", help="Disable preview window during sweep.")
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    profile_path = (project_root / args.profile).resolve()
    base = _read_json(profile_path)
    if not base:
        raise SystemExit(f"failed to read profile: {profile_path}")
    waits = _parse_csv_floats(args.wait_ms)
    targets = _parse_csv_ints(args.target_fps)
    run_id = args.run_id or _now_tag()
    root_out = project_root / "sync_ipc" / "yolo_sweeps" / run_id
    root_out.mkdir(parents=True, exist_ok=True)
    _write_json(root_out / "sweep_config.json", {
        "run_id": run_id,
        "profile": str(profile_path),
        "wait_ms": waits,
        "target_fps": targets,
        "duration_sec": args.duration_sec,
        "display": args.display,
        "disable_bev": bool(args.disable_bev),
        "disable_preview": bool(args.disable_preview),
    })

    results: List[Dict[str, Any]] = []
    for target in targets:
        for wait_ms in waits:
            case_name = f"target{target}_wait{wait_ms:g}ms"
            case_dir = root_out / case_name
            case_profile = case_dir / "profile.json"
            _make_case_profile(base, wait_ms, target, case_profile, args)
            print(f"[sweep] start {case_name}", flush=True)
            t0 = time.time()
            rc = _run_case(project_root, case_profile, case_dir, args.duration_sec, args.display, args.extra_launch_arg)
            summary = _collect_case_artifacts(project_root, case_dir)
            item = {
                "case": case_name,
                "target_fps": target,
                "wait_ms": wait_ms,
                "rc": rc,
                "wall_sec": round(time.time() - t0, 3),
                "summary": summary,
            }
            results.append(item)
            _write_json(root_out / "results.json", {"run_id": run_id, "results": results})
            perf = summary.get("perf", {})
            print(
                "[sweep] done {case} rc={rc} fps={fps} full={full} partial={partial} "
                "collect_avg={cw} total_ms_avg={tm}".format(
                    case=case_name,
                    rc=rc,
                    fps=perf.get("fps_total"),
                    full=perf.get("full_batch_rate"),
                    partial=perf.get("partial_batches"),
                    cw=perf.get("collect_wait_ms_avg"),
                    tm=perf.get("total_ms_avg"),
                ),
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda r: (
            _safe_float(r.get("summary", {}).get("perf", {}).get("fps_total")),
            _safe_float(r.get("summary", {}).get("perf", {}).get("full_batch_rate")),
            -_safe_float(r.get("summary", {}).get("perf", {}).get("collect_wait_ms_avg")),
        ),
        reverse=True,
    )
    _write_json(root_out / "ranked_results.json", {"run_id": run_id, "ranked": ranked})
    if ranked:
        best = ranked[0]
        perf = best.get("summary", {}).get("perf", {})
        print(
            "[sweep] best {case} target={target} wait={wait} fps={fps} full={full}".format(
                case=best.get("case"),
                target=best.get("target_fps"),
                wait=best.get("wait_ms"),
                fps=perf.get("fps_total"),
                full=perf.get("full_batch_rate"),
            ),
            flush=True,
        )
    print(f"[sweep] results={root_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
