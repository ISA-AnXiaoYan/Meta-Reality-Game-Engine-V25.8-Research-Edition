#!/usr/bin/env python3
"""Standalone event overlay overview renderer.

This process intentionally stays separate from the main PyOpenGL/CUDA preview
renderer.  It first tries event_overlay_latest_*.mmap and falls back to the
display-only event_overlay_masked_latest_*.mmap frames, uploads them to OpenGL
textures, applies optional GPU dilation, and renders the diagnostics window
without sharing the main preview Qt event loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from pyopengl_cuda_preview_renderer import (
    QApplication,
    EventOverlayOverviewWindow,
    QSurfaceFormat,
    QTimer,
    Qt,
    _json_safe,
    _parse_cams,
    _place_window,
    _screen_summary,
    _show_and_raise,
)


class _OverlayStateProxy:
    def snapshot(self) -> Dict[str, Any]:
        return {"humans": {}}


class _PreviewProxy:
    def __init__(self, cams: str):
        self.cams = _parse_cams(cams)
        self.event_overlay_stats: Dict[str, Dict[str, Any]] = {
            c: {
                "drawn": False,
                "stale": True,
                "age_ms": -1.0,
                "mode": "nearest_mvs",
                "reason": "not_initialized",
            }
            for c in self.cams
        }
        self.last_frame_ids: Dict[str, int] = {c: -1 for c in self.cams}
        self.overlay_state = _OverlayStateProxy()
        self.event_overlay_overview_window = None

    def set_event_overlay_overview_window(self, window: Any) -> None:
        self.event_overlay_overview_window = window


def _resolve_path(path: str) -> str:
    p = Path(str(path or "")).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def _control_visible(obj: Dict[str, Any], default: bool) -> bool:
    for key in ("event_overlay_overview_visible", "visible", "event_overlay_preview_enable"):
        if key in obj:
            return bool(obj.get(key))
    return bool(default)


def _read_control(path: str, default_visible: bool) -> Dict[str, Any]:
    control_path = _resolve_path(path)
    out: Dict[str, Any] = {
        "path": control_path,
        "visible_requested": bool(default_visible),
        "last_error": "",
    }
    if not path:
        return out
    try:
        p = Path(control_path)
        if not p.exists():
            return out
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            out.update({
                "visible_requested": _control_visible(obj, default_visible),
                "seq": obj.get("seq"),
                "source": obj.get("source", ""),
                "updated_wall_us": obj.get("updated_wall_us", 0),
            })
    except Exception as exc:
        out["last_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _apply_control(window: EventOverlayOverviewWindow, args: argparse.Namespace, control_state: Dict[str, Any], force: bool = False) -> None:
    state = _read_control(str(getattr(args, "event_overlay_overview_control_json", "") or ""), bool(getattr(args, "event_overlay_preview_enable", True)))
    control_state.clear()
    control_state.update(state)
    visible = bool(state.get("visible_requested", True))
    setattr(args, "event_overlay_preview_enable", bool(visible))
    if visible:
        if force or not bool(window.isVisible()):
            _show_and_raise(window, label="event-overlay-overview", force_raise=not bool(args.no_force_raise_windows))
    else:
        if force or bool(window.isVisible()):
            window.hide()


def _write_status(path: str, preview: _PreviewProxy, window: EventOverlayOverviewWindow, args: argparse.Namespace, control_state: Dict[str, Any]) -> None:
    if not path:
        return
    status_path = _resolve_path(path)
    payload = {
        "schema_version": 1,
        "kind": "event_overlay_overview_status",
        "pid": os.getpid(),
        "ts_wall_us": int(time.time() * 1_000_000),
        "cams": list(preview.cams),
        "window": {
            "visible": bool(window.isVisible()),
            "width": int(window.width()),
            "height": int(window.height()),
            "target_fps": float(getattr(args, "event_overlay_window_fps", 30.0) or 30.0),
            "paint_fps": float(getattr(window, "paint_fps_current", 0.0) or 0.0),
            "paint_ms": float(getattr(window, "paint_ms_current", 0.0) or 0.0),
            "paint_interval_ms": float(getattr(window, "paint_interval_ms", 0.0) or 0.0),
        },
        "control": dict(control_state),
        "dilate": {
            "requested_backend": str(getattr(args, "event_overlay_preview_dilate_backend", "gl_shader") or "gl_shader"),
            "requested_px": int(getattr(args, "event_overlay_preview_dilate_px", 4) or 0),
            "effective_backend": str(window._active_dilate_backend()),
            "shader_ready": bool(int(getattr(window, "overlay_shader_program", 0) or 0) > 0),
            "shader_error": str(getattr(window, "overlay_shader_error", "") or ""),
        },
        "event_overlay": preview.event_overlay_stats,
    }
    tmp = status_path + ".tmp"
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
    os.replace(tmp, status_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone event overlay overview renderer")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--vsync", type=int, default=0)
    ap.add_argument("--layout", default="grid2x4")
    ap.add_argument("--no-force-raise-windows", action="store_true")
    ap.add_argument("--event-overlay-preview-enable", action="store_true", default=True)
    ap.add_argument("--event-overlay-window-width", type=int, default=1280)
    ap.add_argument("--event-overlay-window-height", type=int, default=720)
    ap.add_argument("--event-overlay-window-x", type=int, default=-100000)
    ap.add_argument("--event-overlay-window-y", type=int, default=-100000)
    ap.add_argument("--event-overlay-window-fps", type=float, default=30.0)
    ap.add_argument("--event-overlay-window-label-fps", type=float, default=10.0)
    ap.add_argument("--event-overlay-window-font-px", type=int, default=12)
    ap.add_argument("--event-overlay-window-layout", default="grid2x4")
    ap.add_argument("--event-overlay-preview-template", default="event_overlay_latest_{camera},event_overlay_masked_latest_{camera}")
    ap.add_argument("--event-overlay-sidecar-template", default="event_overlay_{camera}_latest.json,event_overlay_masked_{camera}_latest.json")
    ap.add_argument("--event-overlay-preview-alpha", type=float, default=0.55)
    ap.add_argument("--event-overlay-preview-gain", type=float, default=3.0)
    ap.add_argument("--event-overlay-preview-gamma", type=float, default=0.65)
    ap.add_argument("--event-overlay-preview-min-alpha", type=float, default=0.35)
    ap.add_argument("--event-overlay-preview-dilate-px", type=int, default=4)
    ap.add_argument("--event-overlay-preview-dilate-max-pixels", type=int, default=5000)
    ap.add_argument("--event-overlay-preview-dilate-backend", choices=["gl_shader", "cpu_sparse", "off", "auto"], default="gl_shader")
    ap.add_argument("--event-overlay-stale-alpha-scale", type=float, default=0.25)
    ap.add_argument("--event-overlay-preview-max-age-ms", type=float, default=100.0)
    ap.add_argument("--event-overlay-sync-mode", choices=["latest", "nearest_mvs", "nearest_human"], default="nearest_mvs")
    ap.add_argument("--event-overlay-max-sync-delta-ms", type=float, default=50.0)
    ap.add_argument("--event-overlay-hide-stale", action="store_true")
    ap.add_argument("--event-overlay-show-stats", action="store_true", default=True)
    ap.add_argument("--event-overlay-overview-status-json", default="sync_ipc/event_overlay_overview_status.json")
    ap.add_argument("--event-overlay-overview-control-json", default="sync_ipc/event_overlay_overview_control.json")
    ap.add_argument("--event-overlay-overview-status-interval-ms", type=float, default=1000.0)
    args = ap.parse_args()

    if not _parse_cams(args.cams):
        raise SystemExit("--cams is empty")

    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(int(args.vsync))
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    print(f"[event_overlay_overview] QApplication created; Qt screens: {_screen_summary(app)}", flush=True)

    preview = _PreviewProxy(args.cams)
    window = EventOverlayOverviewWindow(preview, args)
    preview.set_event_overlay_overview_window(window)
    _place_window(
        window,
        app,
        preferred_x=int(getattr(args, "event_overlay_window_x", -100000)),
        preferred_y=int(getattr(args, "event_overlay_window_y", -100000)),
        preferred_w=int(getattr(args, "event_overlay_window_width", 1280)),
        preferred_h=int(getattr(args, "event_overlay_window_height", 720)),
        label="event-overlay-overview",
    )
    control_state: Dict[str, Any] = {}
    _apply_control(window, args, control_state, force=True)
    print(f"[event_overlay_overview] window {'shown' if window.isVisible() else 'hidden'}", flush=True)

    control_timer = QTimer()
    control_timer.setTimerType(Qt.PreciseTimer)

    def apply_control() -> None:
        try:
            _apply_control(window, args, control_state)
        except Exception as exc:
            control_state["last_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[event_overlay_overview][WARN] control apply failed: {type(exc).__name__}: {exc}", flush=True)

    control_timer.timeout.connect(apply_control)
    control_timer.start(250)

    status_timer = QTimer()
    status_timer.setTimerType(Qt.PreciseTimer)

    def write_status() -> None:
        try:
            _write_status(str(args.event_overlay_overview_status_json), preview, window, args, control_state)
        except Exception as exc:
            print(f"[event_overlay_overview][WARN] status write failed: {type(exc).__name__}: {exc}", flush=True)

    status_timer.timeout.connect(write_status)
    status_timer.start(max(250, int(float(args.event_overlay_overview_status_interval_ms or 1000.0))))
    QTimer.singleShot(250, write_status)

    print("[event_overlay_overview] entering Qt event loop", flush=True)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
