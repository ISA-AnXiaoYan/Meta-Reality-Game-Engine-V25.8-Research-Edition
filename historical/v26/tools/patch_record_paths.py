#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_record_paths.py

Fix one-key recording path split-brain in launch_fusion_system.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path.cwd().resolve()
LAUNCH = PROJECT_ROOT / "launch_fusion_system.py"
CONFIG = PROJECT_ROOT / "ids_5cam_fusion_config.json"

MARK_BEGIN = "    # ---- YSXQ one-key record path fix begin ----\n"
MARK_END = "    # ---- YSXQ one-key record path fix end ----\n"

INSERT_BLOCK = (
    MARK_BEGIN
    + "    # Force all children to use the same stable absolute command/status files.\n"
    + "    # This avoids cwd-dependent split-brain:\n"
    + "    #   GL cwd = sync_ipc/_runtime_config\n"
    + "    #   EVENT/MVS cwd = project root\n"
    + "    record_cmd_path = str((project_root / 'sync_ipc' / 'record_control.json').resolve())\n"
    + "    record_status_path = str((project_root / 'sync_ipc' / 'record_status.json').resolve())\n"
    + "\n"
    + "    shared = event_cfg.setdefault('shared_options', {})\n"
    + "    shared['enable_remote_raw_record_control'] = True\n"
    + "    shared['raw_record_command_file'] = record_cmd_path\n"
    + "\n"
    + "    one_key_record = mvs_cfg.setdefault('one_key_record', {})\n"
    + "    one_key_record['enable'] = True\n"
    + "    one_key_record['command_file'] = record_cmd_path\n"
    + "    one_key_record['status_file'] = record_status_path\n"
    + MARK_END
)


def patch_launch() -> None:
    if not LAUNCH.exists():
        raise SystemExit(f"[ERROR] not found: {LAUNCH}")

    text = LAUNCH.read_text(encoding="utf-8")

    if MARK_BEGIN in text and MARK_END in text:
        before = text.split(MARK_BEGIN)[0]
        after = text.split(MARK_END, 1)[1]
        new_text = before + INSERT_BLOCK + after
    else:
        needle = "    # Ensure the event-side non-blocking human mask prefilter reads the same MVS\n"
        if needle not in text:
            raise SystemExit(
                "[ERROR] cannot find insertion point in launch_fusion_system.py.\n"
                "Please send me the current launch_fusion_system.py."
            )
        new_text = text.replace(needle, INSERT_BLOCK + "\n" + needle, 1)

    backup = LAUNCH.with_suffix(LAUNCH.suffix + ".bak_record_path")
    if not backup.exists():
        shutil.copy2(LAUNCH, backup)
        print(f"[OK] backup -> {backup}")

    LAUNCH.write_text(new_text, encoding="utf-8")
    print(f"[OK] patched -> {LAUNCH}")


def patch_root_config() -> None:
    if not CONFIG.exists():
        print(f"[WARN] config not found, skip: {CONFIG}")
        return

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    record_cmd = str((PROJECT_ROOT / "sync_ipc" / "record_control.json").resolve())
    record_status = str((PROJECT_ROOT / "sync_ipc" / "record_status.json").resolve())

    event_cfg = cfg.setdefault("event_config", {})
    shared = event_cfg.setdefault("shared_options", {})
    shared["enable_remote_raw_record_control"] = True
    shared["raw_record_command_file"] = record_cmd

    mvs_cfg = cfg.setdefault("mvs_config", {})
    one_key = mvs_cfg.setdefault("one_key_record", {})
    one_key["enable"] = True
    one_key["command_file"] = record_cmd
    one_key["status_file"] = record_status

    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] config record_control -> {record_cmd}")
    print(f"[OK] config record_status  -> {record_status}")


def clean_runtime() -> None:
    runtime = PROJECT_ROOT / "sync_ipc" / "_runtime_config"
    for p in [
        PROJECT_ROOT / "sync_ipc" / "record_control.json",
        PROJECT_ROOT / "sync_ipc" / "record_status.json",
        runtime / "sync_ipc" / "record_control.json",
        runtime / "sync_ipc" / "record_status.json",
    ]:
        try:
            if p.exists() or p.is_symlink():
                p.unlink()
                print(f"[OK] removed {p}")
        except Exception as exc:
            print(f"[WARN] cannot remove {p}: {exc}")

    if runtime.exists():
        shutil.rmtree(runtime)
        print(f"[OK] removed runtime dir -> {runtime}")


def main() -> int:
    print(f"[INFO] project root = {PROJECT_ROOT}")
    patch_launch()
    patch_root_config()
    clean_runtime()
    print("\n[DONE]")
    print("Now restart launch_fusion_system.py, then verify:")
    print("  grep -R \"record_control.json\" sync_ipc/_runtime_config")
    print("Expected: all paths should be /home/.../test607/sync_ipc/record_control.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
