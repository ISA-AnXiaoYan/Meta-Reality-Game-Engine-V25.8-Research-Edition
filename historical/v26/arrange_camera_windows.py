#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arrange event-camera and MVS preview windows in matched A/B/C/... order.

Default layout:
  top row    : event A, event B, ...
  bottom row : MVS   A, MVS   B, ...

Requires X11 and wmctrl:
  sudo apt install wmctrl
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_EVENT_CONFIG = "cameras_sync_live_D_v3_ts_align.json"
DEFAULT_HUMAN_CONFIG = "rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json"
DEFAULT_LAUNCHER = "multi_camera_launcher.py"


@dataclass
class WindowInfo:
    wid: str
    desktop: str
    x: int
    y: int
    w: int
    h: int
    host: str
    title: str


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


def _index_to_alias(index: int) -> str:
    index = int(index)
    chars: List[str] = []
    while True:
        index, rem = divmod(index, 26)
        chars.append(chr(ord("A") + rem))
        if index == 0:
            break
        index -= 1
    return "".join(reversed(chars))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object")
    return obj


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_human_config(path: Path) -> Dict[str, Any]:
    cfg = _load_json(path)
    active = str(cfg.get("active_profile", "") or "").strip()
    if not active:
        return cfg
    profiles = cfg.get("profiles", {}) or {}
    if not isinstance(profiles, dict) or active not in profiles:
        return cfg
    merged = copy.deepcopy(cfg)
    profile_cfg = profiles.get(active) or {}
    if isinstance(profile_cfg, dict):
        _deep_update(merged, profile_cfg)
    merged["active_profile"] = active
    return merged


def event_aliases(path: Path) -> List[str]:
    cfg = _load_json(path)
    aliases: List[str] = []
    for idx, cam in enumerate(cfg.get("cameras", []) or []):
        if not isinstance(cam, dict) or bool(cam.get("enabled", True)) is False:
            continue
        alias = str(cam.get("alias", "") or "").strip()
        aliases.append(alias or _index_to_alias(idx))
    return aliases


def human_aliases(path: Path) -> Tuple[List[str], str]:
    cfg = load_human_config(path)
    aliases: List[str] = []
    for route in cfg.get("routes", []) or []:
        if not isinstance(route, dict) or bool(route.get("enabled", True)) is False:
            continue
        if str(route.get("source", "") or "hik").lower() != "hik":
            continue
        alias = str(route.get("id", route.get("name", "")) or "").strip()
        if alias:
            aliases.append(alias)
    display = cfg.get("display", {}) or {}
    prefix = str(display.get("window_prefix", "RGB_YOLO_MVS_PREVIEW") or "RGB_YOLO_MVS_PREVIEW")
    return aliases, prefix


def merge_aliases(primary: Iterable[str], secondary: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for alias in list(primary) + list(secondary):
        alias = str(alias).strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def read_event_prefix(launcher_path: Path) -> str:
    try:
        for line in launcher_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ARENA_NAME_PREFIX"):
                _, value = line.split("=", 1)
                return str(ast.literal_eval(value.strip()))
    except Exception:
        pass
    return "一时兴起竞技场"


def run_text(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return proc.stdout


def ensure_wmctrl() -> None:
    if shutil.which("wmctrl") is None:
        raise RuntimeError("wmctrl not found. Install it with: sudo apt install wmctrl")


def work_area() -> Rect:
    text = run_text(["wmctrl", "-d"])
    active = ""
    for line in text.splitlines():
        if "*" in line:
            active = line
            break
    active = active or (text.splitlines()[0] if text.splitlines() else "")
    m = re.search(r"WA:\s*(-?\d+),(-?\d+)\s+(\d+)x(\d+)", active)
    if m:
        return Rect(*(int(v) for v in m.groups()))
    m = re.search(r"DG:\s*(\d+)x(\d+)", active)
    if m:
        return Rect(0, 0, int(m.group(1)), int(m.group(2)))
    return Rect(0, 0, 1920, 1080)


def list_windows() -> List[WindowInfo]:
    text = run_text(["wmctrl", "-lG"])
    windows: List[WindowInfo] = []
    for line in text.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        wid, desktop, x, y, w, h, host, title = parts
        try:
            windows.append(WindowInfo(wid, desktop, int(x), int(y), int(w), int(h), host, title))
        except ValueError:
            continue
    return windows


def find_window(windows: List[WindowInfo], title: str, used: set) -> Optional[WindowInfo]:
    title_norm = title.strip()
    candidates = [w for w in windows if w.wid not in used and w.title.strip() == title_norm]
    if not candidates:
        candidates = [w for w in windows if w.wid not in used and title_norm in w.title]
    if not candidates:
        return None
    candidates.sort(key=lambda w: (len(w.title), w.title))
    used.add(candidates[0].wid)
    return candidates[0]


def split_tracks(start: int, total: int, count: int, gap: int) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    usable = max(1, total - gap * (count - 1))
    base = usable // count
    rem = usable % count
    tracks: List[Tuple[int, int]] = []
    cursor = start
    for idx in range(count):
        size = base + (1 if idx < rem else 0)
        tracks.append((cursor, size))
        cursor += size + gap
    return tracks


def build_rects(aliases: List[str], area: Rect, mode: str, margin: int, gap: int) -> Dict[Tuple[str, str], Rect]:
    inner = Rect(
        area.x + margin,
        area.y + margin,
        max(1, area.w - margin * 2),
        max(1, area.h - margin * 2),
    )
    rects: Dict[Tuple[str, str], Rect] = {}
    if mode == "pair-columns":
        rows = split_tracks(inner.y, inner.h, len(aliases), gap)
        cols = split_tracks(inner.x, inner.w, 2, gap)
        for idx, alias in enumerate(aliases):
            y, h = rows[idx]
            rects[("event", alias)] = Rect(cols[0][0], y, cols[0][1], h)
            rects[("mvs", alias)] = Rect(cols[1][0], y, cols[1][1], h)
        return rects

    cols = split_tracks(inner.x, inner.w, len(aliases), gap)
    rows = split_tracks(inner.y, inner.h, 2, gap)
    for idx, alias in enumerate(aliases):
        x, w = cols[idx]
        rects[("event", alias)] = Rect(x, rows[0][0], w, rows[0][1])
        rects[("mvs", alias)] = Rect(x, rows[1][0], w, rows[1][1])
    return rects


def move_window(win: WindowInfo, rect: Rect, dry_run: bool = False) -> None:
    print(f"{win.wid} -> {rect.x},{rect.y},{rect.w},{rect.h}  {win.title}")
    if dry_run:
        return
    subprocess.run(["wmctrl", "-ir", win.wid, "-b", "remove,maximized_vert,maximized_horz"], check=False)
    subprocess.run(["wmctrl", "-ir", win.wid, "-e", f"0,{rect.x},{rect.y},{rect.w},{rect.h}"], check=False)


def arrange_once(args: argparse.Namespace, aliases: List[str], event_prefix: str, mvs_prefix: str) -> bool:
    area = work_area()
    area = Rect(
        int(args.x if args.x is not None else area.x),
        int(args.y if args.y is not None else area.y),
        int(args.width if args.width is not None else area.w),
        int(args.height if args.height is not None else area.h),
    )
    rects = build_rects(aliases, area, args.mode, args.margin, args.gap)
    windows = list_windows()
    used = set()
    missing: List[str] = []
    matches: List[Tuple[WindowInfo, Rect]] = []

    for alias in aliases:
        event_title = args.event_title_template.format(alias=alias, prefix=event_prefix)
        mvs_title = args.mvs_title_template.format(alias=alias, prefix=mvs_prefix)

        event_win = find_window(windows, event_title, used)
        if event_win is None:
            missing.append(event_title)
        else:
            matches.append((event_win, rects[("event", alias)]))

        mvs_win = find_window(windows, mvs_title, used)
        if mvs_win is None:
            missing.append(mvs_title)
        else:
            matches.append((mvs_win, rects[("mvs", alias)]))

    for win, rect in matches:
        move_window(win, rect, dry_run=args.dry_run)

    if missing:
        print("[missing]")
        for title in missing:
            print(f"  {title}")
    return not missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arrange matched event-camera and MVS preview windows.")
    p.add_argument("--event-config", default=DEFAULT_EVENT_CONFIG)
    p.add_argument("--human-config", default=DEFAULT_HUMAN_CONFIG)
    p.add_argument("--launcher", default=DEFAULT_LAUNCHER)
    p.add_argument("--aliases", default="", help="Comma-separated aliases, for example A,B,C,D,E. Defaults to JSON config order.")
    p.add_argument("--mode", choices=["two-rows", "pair-columns"], default="two-rows")
    p.add_argument("--event-title-template", default="{prefix}{alias}")
    p.add_argument("--mvs-title-template", default="{prefix}_{alias}")
    p.add_argument("--timeout", type=float, default=20.0, help="Seconds to wait for all windows before arranging found ones.")
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--watch", action="store_true", help="Keep enforcing the layout until Ctrl+C.")
    p.add_argument("--margin", type=int, default=8)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit nonzero if any expected window is missing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_wmctrl()

    root = Path.cwd()
    event_config = (root / args.event_config).resolve()
    human_config = (root / args.human_config).resolve()
    launcher = (root / args.launcher).resolve()

    if args.aliases.strip():
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
        _, mvs_prefix = human_aliases(human_config)
    else:
        ev_aliases = event_aliases(event_config)
        hv_aliases, mvs_prefix = human_aliases(human_config)
        aliases = merge_aliases(ev_aliases, hv_aliases)

    if not aliases:
        raise RuntimeError("No camera aliases found.")

    event_prefix = read_event_prefix(launcher)
    deadline = time.time() + max(0.0, args.timeout)
    ok = False

    while True:
        ok = arrange_once(args, aliases, event_prefix, mvs_prefix)
        if ok or time.time() >= deadline or args.timeout <= 0:
            break
        time.sleep(max(0.05, args.poll))

    if args.watch:
        try:
            while True:
                time.sleep(max(0.05, args.poll))
                ok = arrange_once(args, aliases, event_prefix, mvs_prefix) and ok
        except KeyboardInterrupt:
            print("\n[window-layout] stopped.")

    return 0 if ok or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
