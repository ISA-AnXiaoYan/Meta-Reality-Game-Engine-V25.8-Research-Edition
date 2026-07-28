#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst = dst_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst_dir / src.name)


def build_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Global Bullet Reprocess Idle Test",
        "",
        "## Verdict",
        "",
        f"- run_id: {summary['run_id']}",
        f"- returncode: {summary['returncode']}",
        f"- idle_test_passed: {summary['idle_test_passed']}",
        f"- duration_sec: {summary['duration_sec']}",
        f"- raw_replay_pack_ready: {summary['status'].get('raw_replay_pack_ready') if isinstance(summary.get('status'), dict) else None}",
        f"- decision_authority: {summary['status'].get('decision_authority') if isinstance(summary.get('status'), dict) else None}",
        f"- publish_to_official_hit_judge: {summary['status'].get('publish_to_official_hit_judge') if isinstance(summary.get('status'), dict) else None}",
        "",
        "## Expected For Empty Load",
        "",
        "- service starts and exits cleanly",
        "- no official hit judge output is written",
        "- status files are produced",
        "- raw replay pack may be absent or empty",
        "",
        "## Artifacts",
        "",
        "- service_stdout.txt",
        "- service_stderr.txt",
        "- global_bullet_reprocess_status.json",
        "- global_bullet_reprocess_latest.json",
        "- global_bullet_reprocess/run_report.json",
    ]
    (run_dir / "idle_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    run_id = args.run_id or time.strftime("global_bullet_reprocess_idle_%Y%m%d_%H%M%S")
    sync_root = Path(args.sync_root)
    if not sync_root.is_absolute():
        sync_root = PROJECT_ROOT / sync_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    run_dir = out_root / run_id
    service_out = sync_root / "global_bullet_reprocess"
    run_dir.mkdir(parents=True, exist_ok=True)
    sync_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "global_bullet_reprocess_service.py"),
        "--project-root",
        str(PROJECT_ROOT),
        "--sync-root",
        str(sync_root),
        "--output-dir",
        str(service_out),
        "--run-id",
        run_id,
        "--mode",
        "idle",
        "--duration-sec",
        str(float(args.duration_sec)),
        "--interval-sec",
        str(float(args.interval_sec)),
        "--allow-empty",
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    elapsed = round(time.time() - started, 3)

    (run_dir / "service_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "service_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    write_json(run_dir / "command.json", {"cmd": cmd, "cwd": str(PROJECT_ROOT)})

    copy_if_exists(sync_root / "global_bullet_reprocess_status.json", run_dir)
    copy_if_exists(sync_root / "global_bullet_reprocess_latest.json", run_dir)
    copy_if_exists(service_out, run_dir)

    status = read_json(sync_root / "global_bullet_reprocess_status.json", {})
    idle_test_passed = (
        proc.returncode == 0
        and isinstance(status, dict)
        and status.get("service") == "global_bullet_reprocess_service"
        and status.get("decision_authority") == "offline_diagnostic_only"
        and status.get("publish_to_official_hit_judge") is False
        and status.get("phase") == "finished"
    )
    summary = {
        "run_id": run_id,
        "returncode": proc.returncode,
        "idle_test_passed": bool(idle_test_passed),
        "duration_sec": elapsed,
        "run_dir": str(run_dir),
        "status": status,
    }
    write_json(run_dir / "idle_test_summary.json", summary)
    build_report(run_dir, summary)
    if args.make_archive:
        archive_base = str(run_dir)
        shutil.make_archive(archive_base, "gztar", root_dir=str(out_root), base_dir=run_id)
        summary["archive"] = archive_base + ".tar.gz"
        write_json(run_dir / "idle_test_summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), "idle_test_passed": idle_test_passed}, ensure_ascii=False, sort_keys=True))
    return 0 if idle_test_passed else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a no-load idle test for global_bullet_reprocess_service.")
    ap.add_argument("--duration-sec", type=float, default=10.0)
    ap.add_argument("--interval-sec", type=float, default=1.0)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--out-root", default="sync_ipc/full_system_tests")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--make-archive", action="store_true")
    return ap.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
