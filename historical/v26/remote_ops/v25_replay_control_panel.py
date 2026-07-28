#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from remote_ops.check_v25_dataset_manifest import build_report, build_resolved_manifest, launch_extra_argv
except Exception:
    from check_v25_dataset_manifest import build_report, build_resolved_manifest, launch_extra_argv  # type: ignore


MODES = ("live", "event_replay", "mvs_replay", "full_replay")
TIMINGS = ("realtime", "fast")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_project_path(project_root: Path, raw: str) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def build_control_payload(
    project_root: Path,
    dataset_dir: str,
    mode: str,
    replay_timing: str,
    profile: str,
    allow_missing_ext_trigger: bool = False,
) -> Dict[str, Any]:
    mode = mode if mode in MODES else "live"
    timing = replay_timing if replay_timing in TIMINGS else "realtime"
    dataset_path = _resolve_project_path(project_root, dataset_dir) if dataset_dir else Path()
    report: Dict[str, Any] = {}
    resolved: Dict[str, Any] = {}
    extra = ""
    extra_argv: list[str] = []
    state = "live_ready" if mode == "live" else "missing_dataset"
    error = ""

    try:
        if mode != "live":
            if not dataset_path or not dataset_path.exists():
                raise FileNotFoundError(f"dataset dir not found: {dataset_path}")
            report = build_report(dataset_path, mode)
            resolved = build_resolved_manifest(
                report,
                replay_timing=timing,
                allow_missing_ext_trigger=allow_missing_ext_trigger,
                event_loop=True,
                mvs_loop=True,
            )
            extra_argv = launch_extra_argv(
                report,
                mode,
                replay_timing=timing,
                event_loop=True,
                mvs_loop=True,
                allow_missing_ext_trigger=allow_missing_ext_trigger,
            )
            if not extra_argv:
                raise RuntimeError(f"no launch arguments resolved for mode={mode}")
            caps = resolved.get("capabilities", {}) if isinstance(resolved.get("capabilities"), dict) else {}
            sync_ok = bool(caps.get("full_replay_sync", True)) if mode == "full_replay" else True
            usable_key = {
                "event_replay": "event_file_replay",
                "mvs_replay": "mvs_file_replay",
                "full_replay": "full_replay",
            }.get(mode, "")
            usable = bool(caps.get(usable_key, True)) if usable_key else True
            state = "ok" if usable and sync_ok else ("degraded" if usable else "not_usable")
        if mode == "live":
            extra_argv = [
                "--event-source-mode", "live_hal",
                "--mvs-source-mode", "live_camera",
            ]
        elif mode == "event_replay":
            extra_argv += ["--mvs-source-mode", "live_camera"]
        elif mode == "mvs_replay":
            extra_argv += ["--event-source-mode", "live_hal"]
        extra = " ".join(extra_argv)
        launch_parts = [sys.executable or "python3", "launch_fusion_system.py", "--launch-profile", profile, *extra_argv]
        launch_command = shlex.join(launch_parts)
    except Exception as exc:
        state = "error"
        error = f"{type(exc).__name__}: {exc}"
        launch_command = ""

    return {
        "schema_version": 1,
        "kind": "v25_replay_control_status",
        "ts_wall_us": int(time.time() * 1_000_000),
        "state": state,
        "error": error,
        "project_root": str(project_root.resolve()),
        "profile": profile,
        "mode": mode,
        "dataset_dir": str(dataset_path) if dataset_dir else "",
        "replay_timing": timing,
        "allow_missing_ext_trigger": bool(allow_missing_ext_trigger),
        "launch_extra_args": extra,
        "launch_extra_argv": extra_argv,
        "launch_command": launch_command,
        "report": {
            "dataset_id": report.get("dataset_id"),
            "session_id": report.get("session_id"),
            "warnings": report.get("warnings", []),
            "missing": report.get("missing", []),
            "validation": report.get("validation", {}),
            "counts": report.get("counts", {}),
        },
        "resolved_manifest": {
            "dataset_id": resolved.get("dataset_id"),
            "session_id": resolved.get("session_id"),
            "scenario": resolved.get("scenario"),
            "capabilities": resolved.get("capabilities"),
            "degraded_reasons": resolved.get("degraded_reasons", []),
            "recommended_launch": resolved.get("recommended_launch", {}),
        },
    }


def write_control_files(project_root: Path, payload: Dict[str, Any]) -> Dict[str, str]:
    sync_dir = project_root / "sync_ipc"
    control_path = sync_dir / "v25_replay_control.json"
    status_path = sync_dir / "v25_replay_status.json"
    command_path = sync_dir / "v25_replay_launch_command.sh"
    _write_json(control_path, {
        "schema_version": 1,
        "kind": "v25_replay_control",
        "ts_wall_us": payload.get("ts_wall_us"),
        "profile": payload.get("profile"),
        "mode": payload.get("mode"),
        "dataset_dir": payload.get("dataset_dir"),
        "replay_timing": payload.get("replay_timing"),
        "allow_missing_ext_trigger": payload.get("allow_missing_ext_trigger"),
        "launch_extra_args": payload.get("launch_extra_args"),
        "launch_extra_argv": payload.get("launch_extra_argv", []),
    })
    _write_json(status_path, payload)
    command = str(payload.get("launch_command") or "")
    command_path.write_text(command + "\n", encoding="utf-8")
    dataset_dir = str(payload.get("dataset_dir") or "")
    resolved = payload.get("resolved_manifest") if isinstance(payload.get("resolved_manifest"), dict) else {}
    if dataset_dir and resolved.get("dataset_id"):
        resolved_path = Path(dataset_dir) / "replay_manifest_resolved_from_panel.json"
        _write_json(resolved_path, payload)
    return {
        "control": str(control_path.resolve()),
        "status": str(status_path.resolve()),
        "command": str(command_path.resolve()),
    }


def _console_loop(args: argparse.Namespace, project_root: Path) -> int:
    payload = build_control_payload(
        project_root,
        args.dataset_dir,
        args.mode,
        args.replay_timing,
        args.profile,
        args.allow_missing_ext_trigger,
    )
    outputs = write_control_files(project_root, payload) if args.write else {}
    if args.print_json or not args.write:
        print(json.dumps({"payload": payload, "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0 if payload.get("state") in {"ok", "degraded", "live_ready"} else 2


def _gui_loop(args: argparse.Namespace, project_root: Path) -> int:
    try:
        from PySide6.QtCore import QProcess, Qt
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QPlainTextEdit,
            QWidget,
        )
    except Exception:
        return _console_loop(args, project_root)

    app = QApplication(sys.argv[:1])
    win = QWidget()
    win.setWindowTitle("元现实游戏引擎 - 启动控制面板")
    win.resize(1080, 620)
    layout = QGridLayout(win)

    profile_edit = QLineEdit(args.profile)
    dataset_edit = QLineEdit(args.dataset_dir)
    mode_box = QComboBox()
    mode_box.addItems(list(MODES))
    mode_box.setCurrentText(args.mode if args.mode in MODES else "live")
    timing_box = QComboBox()
    timing_box.addItems(list(TIMINGS))
    timing_box.setCurrentText(args.replay_timing if args.replay_timing in TIMINGS else "realtime")
    allow_box = QCheckBox("允许缺失 EXTTRIG（降级调试）")
    allow_box.setChecked(bool(args.allow_missing_ext_trigger))
    output = QPlainTextEdit()
    output.setReadOnly(True)
    output.setFont(QFont("DejaVu Sans Mono", 10))

    def choose_dataset() -> None:
        path = QFileDialog.getExistingDirectory(win, "选择 V25 数据集目录", dataset_edit.text() or str(project_root))
        if path:
            dataset_edit.setText(path)

    launch_state = QLabel("系统状态：尚未从面板启动")
    launch_process = QProcess(win)
    launch_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

    def current_payload() -> Dict[str, Any]:
        return build_control_payload(
            project_root,
            dataset_edit.text().strip(),
            mode_box.currentText(),
            timing_box.currentText(),
            profile_edit.text().strip(),
            allow_box.isChecked(),
        )

    def refresh(write_files: bool) -> None:
        payload = current_payload()
        outputs = write_control_files(project_root, payload) if write_files else {}
        output.setPlainText(json.dumps({"payload": payload, "outputs": outputs}, ensure_ascii=False, indent=2))

    def drain_launch_output() -> None:
        chunk = bytes(launch_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if chunk:
            output.appendPlainText(chunk.rstrip())

    def launch_finished(exit_code: int, _exit_status: Any) -> None:
        drain_launch_output()
        launch_btn.setEnabled(True)
        status_path = project_root / "sync_ipc" / "v25_panel_launch_status.json"
        try:
            launch_status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            launch_status = {"state": "unknown", "error": f"launch_status_read_failed: {exc}"}
        state = str(launch_status.get("state") or "unknown")
        launch_state.setText(f"系统状态：{state}（进程返回 {exit_code}）")
        output.appendPlainText(json.dumps({"panel_launch_status": launch_status}, ensure_ascii=False, indent=2))

    def launch_error(error: Any) -> None:
        launch_btn.setEnabled(True)
        launch_state.setText(f"系统状态：启动器调用失败（{error}）")

    def start_full_system() -> None:
        if launch_process.state() != QProcess.ProcessState.NotRunning:
            launch_state.setText("系统状态：启动请求正在处理中")
            return
        payload = current_payload()
        outputs = write_control_files(project_root, payload)
        state = str(payload.get("state") or "")
        startable = state in {"live_ready", "ok"} or (state == "degraded" and allow_box.isChecked())
        output.setPlainText(json.dumps({"payload": payload, "outputs": outputs}, ensure_ascii=False, indent=2))
        if not startable:
            launch_state.setText(f"系统状态：未启动，当前配置不可用（{state or 'unknown'}）")
            return
        launch_btn.setEnabled(False)
        launch_state.setText("系统状态：正在启动并等待全链路就绪…")
        launch_process.setWorkingDirectory(str(project_root))
        launch_process.setProgram(sys.executable or "python3")
        launch_process.setArguments([
            "remote_ops/start_system_from_panel.py",
            "--project-root", str(project_root),
            "--control", "sync_ipc/v25_replay_status.json",
            "--status", "sync_ipc/v25_panel_launch_status.json",
            "--label", "desktop_latest",
            "--ready-wait", "120",
        ])
        launch_process.start()

    pick_btn = QPushButton("选择数据集")
    check_btn = QPushButton("校验")
    write_btn = QPushButton("写入控制状态")
    launch_btn = QPushButton("一键启动整套系统")
    pick_btn.clicked.connect(choose_dataset)
    check_btn.clicked.connect(lambda: refresh(False))
    write_btn.clicked.connect(lambda: refresh(True))
    launch_btn.clicked.connect(start_full_system)
    launch_process.readyReadStandardOutput.connect(drain_launch_output)
    launch_process.finished.connect(launch_finished)
    launch_process.errorOccurred.connect(launch_error)

    layout.addWidget(QLabel("启动 Profile"), 0, 0)
    layout.addWidget(profile_edit, 0, 1, 1, 3)
    layout.addWidget(QLabel("回放数据集"), 1, 0)
    layout.addWidget(dataset_edit, 1, 1)
    layout.addWidget(pick_btn, 1, 2)
    row = QHBoxLayout()
    row.addWidget(QLabel("源模式"))
    row.addWidget(mode_box)
    row.addWidget(QLabel("回放节拍"))
    row.addWidget(timing_box)
    row.addWidget(allow_box)
    row.setAlignment(Qt.AlignLeft)
    layout.addLayout(row, 2, 1, 1, 3)
    layout.addWidget(check_btn, 3, 1)
    layout.addWidget(write_btn, 3, 2)
    layout.addWidget(launch_btn, 3, 3)
    layout.addWidget(launch_state, 4, 0, 1, 4)
    layout.addWidget(output, 5, 0, 1, 4)
    refresh(False)
    win.show()
    return int(app.exec())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="V25 replay control panel/status contract helper.")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--profile", default="launch_profiles/627_event_overlay_bev.json")
    ap.add_argument("--dataset-dir", default="")
    ap.add_argument("--mode", choices=MODES, default="live")
    ap.add_argument("--replay-timing", choices=TIMINGS, default="realtime")
    ap.add_argument("--allow-missing-ext-trigger", action="store_true", default=False)
    ap.add_argument("--write", action="store_true", default=False)
    ap.add_argument("--print-json", action="store_true", default=False)
    ap.add_argument("--gui", action="store_true", default=False)
    args = ap.parse_args(argv)
    project_root = _resolve_project_path(Path.cwd(), args.project_root)
    if args.gui:
        return _gui_loop(args, project_root)
    return _console_loop(args, project_root)


if __name__ == "__main__":
    raise SystemExit(main())
