#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export YOLO segmentation model from ids_8cam_fusion_config.json to TensorRT engine.

This version is written for your unified config structure:

    ids_8cam_fusion_config.json
      -> mvs_config
        -> common
          -> model
            -> model: /path/to/yolo11s-seg.pt

It exports the .pt model to .engine and, with --update-config,
updates mvs_config.common.model.model to the generated engine path.
It does not touch MVS sync, event camera, bullet trajectory, hit_judge,
UE5 bridge, camera serials, or BGR8 settings.

Typical use from project root:

    python3 tools/export_yolo_engine_from_ids8_config.py \
      --config ids_8cam_fusion_config.json \
      --batch 8 \
      --imgsz 768 \
      --device 0 \
      --half true \
      --max-det 20 \
      --retina-masks true \
      --update-config

If fixed batch=8 engine has a runtime batch mismatch, re-export dynamic:

    python3 tools/export_yolo_engine_from_ids8_config.py \
      --config ids_8cam_fusion_config.json \
      --batch 8 \
      --imgsz 768 \
      --device 0 \
      --half true \
      --max-det 20 \
      --retina-masks true \
      --dynamic \
      --update-config
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


MODEL_CFG_PATHS: List[Tuple[str, ...]] = [
    ("mvs_config", "common", "model"),   # your ids_8cam_fusion_config.json path
    ("common", "model"),                 # compatibility with old exporter
]


def _sanitize_python_path_for_export() -> None:
    """Avoid Ubuntu system packages shadowing conda packages during ONNX/TensorRT export."""
    clean = []
    removed = []
    for item in sys.path:
        text = str(item)
        if text.startswith("/usr/lib/python3/dist-packages") or text.startswith("/usr/local/lib/python3/dist-packages"):
            removed.append(text)
            continue
        clean.append(item)
    if removed:
        sys.path[:] = clean
        print("[INFO] removed system dist-packages from sys.path:")
        for item in removed:
            print(f"  - {item}")

    for name in list(sys.modules):
        if name == "google" or name.startswith("google.protobuf"):
            sys.modules.pop(name, None)


def _check_export_imports() -> None:
    import google.protobuf
    from google.protobuf.internal import builder
    import onnx
    print(f"[OK] protobuf {google.protobuf.__version__}: {google.protobuf.__file__}")
    print(f"[OK] protobuf builder: {builder.__file__}")
    print(f"[OK] onnx {onnx.__version__}: {onnx.__file__}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return data


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _get_nested(data: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _ensure_nested_dict(data: Dict[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            raise ValueError(f"cannot create nested dict at {'.'.join(keys)}")
        cur = cur.setdefault(key, {})
    if not isinstance(cur, dict):
        raise ValueError(f"{'.'.join(keys)} must be a JSON object")
    return cur


def _find_model_cfg(data: Dict[str, Any]) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    """Find the model config object used by this project."""
    for path in MODEL_CFG_PATHS:
        cfg = _get_nested(data, path)
        if isinstance(cfg, dict) and str(cfg.get("model", "")).strip():
            return path, cfg

    searched = ", ".join(".".join(p) for p in MODEL_CFG_PATHS)
    raise ValueError(
        "Could not find a YOLO model path in config. Expected one of: "
        f"{searched}. For your ids_8cam config it should be "
        "mvs_config.common.model.model"
    )


def _resolve_path(path_text: str, config_path: Path, project_root: Path) -> Path:
    p = Path(str(path_text).strip()).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidates = [
        project_root / p,
        config_path.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (project_root / p).resolve()


def _find_engine(export_result: Any, source_model: Path) -> Path:
    if isinstance(export_result, (str, os.PathLike)):
        p = Path(export_result).expanduser()
        if p.exists():
            return p.resolve()

    expected = source_model.with_suffix(".engine")
    if expected.exists():
        return expected.resolve()

    matches = sorted(
        source_model.parent.glob(source_model.stem + "*.engine"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(f"TensorRT engine was not found near {source_model}")


def _copy_engine_to_project(engine_path: Path, project_root: Path) -> Path:
    target = project_root / engine_path.name
    if engine_path.resolve() != target.resolve():
        shutil.copy2(engine_path, target)
        return target.resolve()
    return engine_path.resolve()


def _path_for_config(engine_path: Path, config_path: Path, absolute: bool) -> str:
    if absolute:
        return str(engine_path.resolve())
    try:
        return os.path.relpath(engine_path.resolve(), config_path.parent.resolve())
    except Exception:
        return str(engine_path.resolve())


def _update_model_cfg(
    cfg: Dict[str, Any],
    engine_text: str,
    max_det: int,
    retina_masks: bool,
    half: bool,
    imgsz: int,
    device: str,
) -> None:
    cfg["model"] = engine_text
    cfg["max_det"] = int(max_det)
    cfg["retina_masks"] = bool(retina_masks)
    cfg["half"] = bool(half)
    cfg["imgsz"] = int(imgsz)
    cfg["device"] = str(device)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export YOLO model in ids_8cam_fusion_config.json to TensorRT engine.")
    ap.add_argument("--config", default="ids_8cam_fusion_config.json", help="unified ids_8cam config JSON")
    ap.add_argument("--batch", type=int, default=8, help="TensorRT export batch size")
    ap.add_argument("--imgsz", type=int, default=0, help="override export imgsz; default reads mvs_config.common.model.imgsz")
    ap.add_argument("--device", default="", help="CUDA device; default reads mvs_config.common.model.device")
    ap.add_argument("--half", default="true", help="true/false")
    ap.add_argument("--max-det", type=int, default=0, help="post-export config max_det; default reads config")
    ap.add_argument("--retina-masks", default="", help="true/false; default reads config")
    ap.add_argument("--dynamic", action="store_true", help="export dynamic TensorRT engine instead of fixed batch")
    ap.add_argument("--update-config", action="store_true", help="update config to point to generated .engine")
    ap.add_argument("--copy-engine-to-project", action="store_true", default=True, help="copy .engine to project root")
    ap.add_argument("--no-copy-engine-to-project", dest="copy_engine_to_project", action="store_false")
    ap.add_argument("--absolute-engine-path", action="store_true", help="write absolute .engine path into config")
    ap.add_argument("--backup", action="store_true", default=True, help="backup config before updating")
    ap.add_argument("--no-backup", dest="backup", action="store_false")
    ap.add_argument(
        "--sync-top-common",
        action="store_true",
        help="also create/update top-level common.model for compatibility with old tools",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    data = _load_json(config_path)
    model_path_keys, model_cfg = _find_model_cfg(data)

    model_text = str(model_cfg.get("model", "")).strip()
    model_path = _resolve_path(model_text, config_path=config_path, project_root=project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    imgsz = int(args.imgsz or model_cfg.get("imgsz", 768) or 768)
    device = str(args.device or model_cfg.get("device", "0") or "0")
    half = _as_bool(args.half, _as_bool(model_cfg.get("half"), True))
    retina_masks = _as_bool(args.retina_masks, _as_bool(model_cfg.get("retina_masks"), True))
    max_det = int(args.max_det or model_cfg.get("max_det", 20) or 20)

    print("========== YOLO TensorRT export for ids_8cam config ==========")
    print(f"project_root={project_root}")
    print(f"config={config_path}")
    print(f"model_cfg_path={'.'.join(model_path_keys)}")
    print(f"source_model={model_path}")
    print(f"imgsz={imgsz} batch={args.batch} half={half} device={device} dynamic={args.dynamic}")
    print(f"post-export config settings: max_det={max_det}, retina_masks={retina_masks}")

    if model_path.suffix.lower() == ".engine":
        engine_path = model_path.resolve()
        print(f"[INFO] config already points to an engine: {engine_path}")
    else:
        _sanitize_python_path_for_export()
        _check_export_imports()

        from ultralytics import YOLO

        yolo = YOLO(str(model_path))
        export_result = yolo.export(
            format="engine",
            imgsz=imgsz,
            batch=int(args.batch),
            half=bool(half),
            device=device,
            dynamic=bool(args.dynamic),
        )
        engine_path = _find_engine(export_result, model_path)
        print(f"[OK] exported engine: {engine_path}")

    if args.copy_engine_to_project:
        engine_path = _copy_engine_to_project(engine_path, project_root)
        print(f"[OK] engine path for config: {engine_path}")

    if args.update_config:
        if args.backup:
            backup_path = config_path.with_suffix(config_path.suffix + ".bak_before_engine_export")
            shutil.copy2(config_path, backup_path)
            print(f"[OK] backup config: {backup_path}")

        engine_text = _path_for_config(engine_path, config_path, absolute=args.absolute_engine_path)

        # Update the actual model config used by your unified MVS config.
        current_cfg = _ensure_nested_dict(data, model_path_keys)
        _update_model_cfg(
            current_cfg,
            engine_text=engine_text,
            max_det=max_det,
            retina_masks=retina_masks,
            half=half,
            imgsz=imgsz,
            device=device,
        )
        print(f"[OK] updated {'.'.join(model_path_keys)}.model -> {engine_text}")

        # Optional compatibility mirror for tools that expect top-level common.model.
        if args.sync_top_common:
            top_cfg = _ensure_nested_dict(data, ("common", "model"))
            mirror = deepcopy(current_cfg)
            data["common"]["model"] = mirror
            print("[OK] synced top-level common.model")

        _save_json(config_path, data)
        print(f"[OK] wrote config: {config_path}")
    else:
        print("[INFO] config not modified. Re-run with --update-config to switch JSON to the engine.")
        print(f"[INFO] engine={engine_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
