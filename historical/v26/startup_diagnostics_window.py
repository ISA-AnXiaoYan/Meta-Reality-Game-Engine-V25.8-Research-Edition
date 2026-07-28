#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as fp:
            obj = json.load(fp)
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        return {"state": "waiting", "title": "Waiting for diagnostics JSON", "message": f"{type(exc).__name__}: {exc}"}


def _format_status(data: Dict[str, Any]) -> str:
    state = str(data.get("state", "unknown"))
    title = str(data.get("title", "EVENT startup diagnostics"))
    msg = str(data.get("message", ""))
    threshold = data.get("usb_threshold_mbps", 0.3)
    lines = [
        f"{title}",
        f"state={state} threshold={threshold}MB/s",
    ]
    if msg:
        lines.append(f"message={msg}")
    if data.get("usb_error"):
        lines.append(f"usb_error={data.get('usb_error')}")
    missing = data.get("missing_cams", [])
    ready = data.get("ready_cams", [])
    lines.append(f"ready={','.join(map(str, ready)) or '-'} missing={','.join(map(str, missing)) or '-'}")
    lines.append("")
    lines.append("cam  ready  by              usb_MB/s  serial               reason")
    lines.append("---  -----  --------------  --------  -------------------  ------------------------------")
    statuses = data.get("camera_status", {})
    if isinstance(statuses, dict):
        for cam in data.get("cams", sorted(statuses.keys())):
            rec = statuses.get(str(cam), {})
            if not isinstance(rec, dict):
                rec = {}
            ready_s = "yes" if rec.get("ready") else "no"
            by_s = str(rec.get("ready_by", ""))[:14]
            usb_s = f"{float(rec.get('usb_mbps', 0.0) or 0.0):.3f}"
            serial_s = str(rec.get("serial", ""))[:19]
            reason_s = str(rec.get("reason", "") or rec.get("recent_event_log_error", ""))[:120]
            lines.append(f"{str(cam):<3}  {ready_s:<5}  {by_s:<14}  {usb_s:>8}  {serial_s:<19}  {reason_s}")
    return "\n".join(lines)


def _console_loop(args: argparse.Namespace) -> int:
    path = Path(args.json)
    ok_since = 0.0
    while True:
        data = _read_json(path)
        print("\033[2J\033[H" + _format_status(data), flush=True)
        if str(data.get("state", "")).lower() == "ok":
            ok_since = ok_since or time.time()
            if (time.time() - ok_since) * 1000.0 >= int(args.auto_exit_ok_ms):
                return 0
        else:
            ok_since = 0.0
        time.sleep(max(0.2, float(args.poll_ms) / 1000.0))


def main() -> int:
    ap = argparse.ArgumentParser(description="EVENT startup diagnostics window")
    ap.add_argument("--json", required=True)
    ap.add_argument("--poll-ms", type=float, default=500.0)
    ap.add_argument("--auto-exit-ok-ms", type=int, default=2500)
    args = ap.parse_args()

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication, QPlainTextEdit, QWidget, QVBoxLayout
    except Exception:
        return _console_loop(args)

    app = QApplication(sys.argv[:1])
    win = QWidget()
    win.setWindowTitle("EVENT Startup Diagnostics")
    win.resize(980, 420)
    layout = QVBoxLayout(win)
    text = QPlainTextEdit()
    text.setReadOnly(True)
    text.setFont(QFont("DejaVu Sans Mono", 11))
    text.setStyleSheet("QPlainTextEdit { color: #c7f7d4; background: #07110c; }")
    layout.addWidget(text)
    win.show()
    ok_since = {"value": 0.0}

    def refresh() -> None:
        data = _read_json(Path(args.json))
        text.setPlainText(_format_status(data))
        state = str(data.get("state", "")).lower()
        if state == "ok":
            if ok_since["value"] <= 0.0:
                ok_since["value"] = time.time()
            if (time.time() - ok_since["value"]) * 1000.0 >= int(args.auto_exit_ok_ms):
                app.quit()
        else:
            ok_since["value"] = 0.0
        if state == "error":
            try:
                win.raise_()
                win.activateWindow()
            except Exception:
                pass

    timer = QTimer(win)
    timer.timeout.connect(refresh)
    timer.start(max(100, int(args.poll_ms)))
    refresh()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
