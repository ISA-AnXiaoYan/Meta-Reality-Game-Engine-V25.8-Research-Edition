#!/usr/bin/env python3
"""Switch event human-mask buffer eval rounds in both config and launch profile."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
from pathlib import Path


BASE_25 = {
    "event_human_mask_buffer_enable": True,
    "event_human_mask_buffer_ms": 25.0,
    "event_human_mask_buffer_timeout_ms": 25.0,
    "event_human_mask_buffer_release_policy": "raw_at_budget",
    "event_human_mask_buffer_max_events": 500000,
    "event_human_mask_buffer_max_buckets": 12,
    "event_human_mask_buffer_empty_release_fps": 5.0,
    "event_human_mask_filter_max_fps": 40.0,
}


def _settings(**updates):
    out = dict(BASE_25)
    out.update(updates)
    return out


ROUND_SETTINGS = {
    "off": _settings(event_human_mask_buffer_enable=False),
    "1": _settings(),
    "b1": _settings(),
    "2": _settings(),
    "b2": _settings(),
    "3": _settings(),
    "b3": _settings(),
    "d1": _settings(event_human_mask_buffer_ms=40.0, event_human_mask_buffer_timeout_ms=40.0),
    "d2": _settings(event_human_mask_buffer_ms=60.0, event_human_mask_buffer_timeout_ms=60.0),
    "d3": _settings(event_human_mask_buffer_ms=90.0, event_human_mask_buffer_timeout_ms=90.0),
    "x1": _settings(
        event_human_mask_buffer_ms=90.0,
        event_human_mask_buffer_timeout_ms=150.0,
        event_human_mask_buffer_release_policy="raw_at_timeout",
    ),
}


def _shared_options(cfg):
    for section in ("event_config", "global_config", "global"):
        obj = cfg.get(section)
        if isinstance(obj, dict) and isinstance(obj.get("shared_options"), dict):
            return obj["shared_options"], section
    event_cfg = cfg.setdefault("event_config", {})
    return event_cfg.setdefault("shared_options", {}), "event_config"


def _cleanup_wrong_global(cfg):
    obj = cfg.get("global")
    if not isinstance(obj, dict) or set(obj.keys()) != {"shared_options"}:
        return False
    opts = obj.get("shared_options")
    if not isinstance(opts, dict):
        return False
    known_keys = set()
    for item in ROUND_SETTINGS.values():
        known_keys.update(item.keys())
    if set(opts.keys()).issubset(known_keys):
        cfg.pop("global", None)
        return True
    return False


def _profile_args(profile_cfg):
    if isinstance(profile_cfg.get("args"), dict):
        return profile_cfg["args"], "args"
    return profile_cfg, ""


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _patch_config(path: Path, settings: dict, round_name: str, no_backup: bool):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    shared, section = _shared_options(cfg)
    cleaned_wrong_global = _cleanup_wrong_global(cfg)
    backup = None
    if not no_backup:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".before_mask_buffer_round_{round_name}_{stamp}.bak")
        shutil.copy2(path, backup)
    shared.update(settings)
    _write_json(path, cfg)
    return {
        "config": str(path),
        "section": section,
        "cleaned_wrong_global": bool(cleaned_wrong_global),
        "backup": str(backup) if backup else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, choices=sorted(ROUND_SETTINGS.keys()))
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "ids_8cam_fusion_config.json"))
    ap.add_argument(
        "--profile",
        default=str(Path(__file__).resolve().parents[1] / "launch_profiles" / "627_event_overlay_bev_v3_sandbox.json"),
        help="Launch profile to patch with the same event mask buffer settings.",
    )
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    profile_path = Path(args.profile) if str(args.profile or "").strip() else None
    settings = ROUND_SETTINGS[str(args.round)]
    profile_section = ""
    profile_backup = None
    profile_updated = False
    profile_cfg = None
    profile_args = None
    config_results = []

    if config_path.exists():
        config_results.append(_patch_config(config_path, settings, str(args.round), bool(args.no_backup)))
    else:
        config_results.append({"config": str(config_path), "exists": False})

    if profile_path:
        if not profile_path.exists():
            raise SystemExit(f"profile not found: {profile_path}")
        if not args.no_backup:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_backup = profile_path.with_suffix(
                profile_path.suffix + f".before_mask_buffer_round_{args.round}_{stamp}.bak"
            )
            shutil.copy2(profile_path, profile_backup)
        profile_cfg = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_args, profile_section = _profile_args(profile_cfg)
        profile_args.update(settings)
        _write_json(profile_path, profile_cfg)
        profile_updated = True
        profile_config = str(profile_args.get("config", "") or "").strip()
        if profile_config:
            profile_config_path = Path(profile_config)
            if not profile_config_path.is_absolute():
                profile_config_path = (profile_path.parent.parent / profile_config_path).resolve()
            already = {
                str(Path(item.get("config", "")).resolve())
                for item in config_results
                if item.get("config") and Path(item.get("config")).exists()
            }
            if str(profile_config_path.resolve()) not in already:
                if profile_config_path.exists():
                    config_results.append(
                        _patch_config(profile_config_path, settings, str(args.round), bool(args.no_backup))
                    )
                else:
                    config_results.append({"config": str(profile_config_path), "exists": False, "source": "profile"})
    print(json.dumps({
        "config": str(config_path),
        "configs": config_results,
        "profile": str(profile_path) if profile_path else "",
        "profile_section": profile_section,
        "profile_updated": bool(profile_updated),
        "round": str(args.round),
        "settings": settings,
        "profile_backup": str(profile_backup) if profile_backup else "",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
