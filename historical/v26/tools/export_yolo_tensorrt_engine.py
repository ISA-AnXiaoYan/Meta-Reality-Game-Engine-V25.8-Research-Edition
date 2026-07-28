#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the configured Ultralytics YOLO model to TensorRT engine.

Typical use from the project root:

    python3 tools/export_yolo_tensorrt_engine.py \
      --config hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json \
      --batch 5 \
      --max-det 20 \
      --retina-masks true \
      --update-configs

This script intentionally keeps the normal JSON using the .pt model until the
.engine file is actually exported on this machine.  With --update-configs it
updates all requested config files to point at the generated .engine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional




def _sanitize_python_path_for_export() -> None:
    """Keep this conda env ahead of Ubuntu system dist-packages during export.

    Ultralytics imports ONNX during TensorRT export.  On some Ubuntu/conda
    mixed environments, /usr/lib/python3/dist-packages can appear in sys.path
    and its old google.protobuf package shadows the conda protobuf package,
    causing: cannot import name 'builder' from google.protobuf.internal.
    This export helper does not need system dist-packages, so remove them.
    """
    removed = []
    clean = []
    for item in sys.path:
        text = str(item)
        if text.startswith("/usr/lib/python3/dist-packages") or text.startswith("/usr/local/lib/python3/dist-packages"):
            removed.append(text)
            continue
        clean.append(item)
    if removed:
        sys.path[:] = clean
        print("[INFO] removed system dist-packages from sys.path for export:")
        for item in removed:
            print(f"  - {item}")

    # If a wrong google/protobuf namespace package was already imported by a
    # site customization hook or earlier dependency, clear it so imports are
    # resolved again from the conda environment.
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


def _model_cfg(data: Dict[str, Any]) -> Dict[str, Any]:
    common = data.setdefault("common", {})
    if not isinstance(common, dict):
        raise ValueError("config.common must be a JSON object")
    model = common.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("config.common.model must be a JSON object")
    return model


def _resolve_path(path_text: str, config_path: Path) -> Path:
    p = Path(str(path_text).strip()).expanduser()
    if p.is_absolute():
        return p
    candidates = [
        Path.cwd() / p,
        config_path.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Prefer project-root relative path for new outputs and error messages.
    return (Path.cwd() / p).resolve()


def _relative_or_abs(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except Exception:
        return str(path.resolve())


def _find_engine(export_result: Any, source_model: Path) -> Path:
    if isinstance(export_result, (str, os.PathLike)):
        p = Path(export_result).expanduser()
        if p.exists():
            return p.resolve()
    expected = source_model.with_suffix(".engine")
    if expected.exists():
        return expected.resolve()
    matches = sorted(source_model.parent.glob(source_model.stem + "*.engine"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"TensorRT engine was not found near {source_model}")


def _copy_engine_to_project(engine_path: Path, project_root: Path) -> Path:
    target = project_root / engine_path.name
    if engine_path.resolve() != target.resolve():
        shutil.copy2(engine_path, target)
        return target.resolve()
    return engine_path.resolve()


def _update_config(path: Path, engine_path: Path, max_det: int, retina_masks: bool, half: bool, copy_to_project: bool) -> None:
    data = _load_json(path)
    m = _model_cfg(data)
    if copy_to_project:
        # Configs in subdirectories should still point to the project-root engine
        # with a relative path that works from that config's directory.
        m["model"] = _relative_or_abs(engine_path, path.parent)
    else:
        m["model"] = str(engine_path)
    m["max_det"] = int(max_det)
    m["retina_masks"] = bool(retina_masks)
    m["half"] = bool(half)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[OK] updated config: {path} -> model={m['model']}, max_det={m['max_det']}, retina_masks={m['retina_masks']}, half={m['half']}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export configured YOLO model to TensorRT engine and optionally update configs.")
    ap.add_argument("--config", default="hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json", help="main config JSON")
    ap.add_argument("--extra-config", action="append", default=[], help="additional config JSON to update; can be used more than once")
    ap.add_argument("--batch", type=int, default=5, help="TensorRT export batch size; use 5 for five cameras")
    ap.add_argument("--imgsz", type=int, default=0, help="override export imgsz; default reads common.model.imgsz")
    ap.add_argument("--device", default="", help="override CUDA device; default reads common.model.device")
    ap.add_argument("--half", default="true", help="true/false, default true")
    ap.add_argument("--max-det", type=int, default=20, help="write this max_det to configs after export")
    ap.add_argument("--retina-masks", default="true", help="true/false; keep true for precise hit-judge masks")
    ap.add_argument("--dynamic", action="store_true", help="export dynamic TensorRT engine instead of fixed batch")
    ap.add_argument("--update-configs", action="store_true", help="update config files to point at the generated engine")
    ap.add_argument("--copy-engine-to-project", action="store_true", default=True, help="copy engine to project root before updating configs")
    ap.add_argument("--no-copy-engine-to-project", dest="copy_engine_to_project", action="store_false")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    data = _load_json(config_path)
    m = _model_cfg(data)

    model_text = str(m.get("model", "") or "").strip()
    if not model_text:
        raise ValueError("common.model.model is empty")
    model_path = _resolve_path(model_text, config_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    imgsz = int(args.imgsz or m.get("imgsz", 768) or 768)
    device = str(args.device or m.get("device", "0") or "0")
    half = _as_bool(args.half, True)
    retina_masks = _as_bool(args.retina_masks, True)
    max_det = max(1, int(args.max_det or m.get("max_det", 20) or 20))

    print("========== TensorRT export ==========")
    print(f"project_root={project_root}")
    print(f"config={config_path}")
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
            half=half,
            device=device,
            dynamic=bool(args.dynamic),
        )
        engine_path = _find_engine(export_result, model_path)
        print(f"[OK] exported engine: {engine_path}")

    if args.copy_engine_to_project:
        engine_path = _copy_engine_to_project(engine_path, project_root)
        print(f"[OK] engine path for configs: {engine_path}")

    if args.update_configs:
        config_paths: List[Path] = [config_path]
        default_extra = [
            project_root / "rentijiance" / config_path.name,
            project_root / "cpp_bullet_core" / "rentijiance" / config_path.name,
        ]
        for p in default_extra:
            if p.exists() and p.resolve() not in {x.resolve() for x in config_paths}:
                config_paths.append(p.resolve())
        for extra in args.extra_config:
            p = Path(extra).expanduser()
            if not p.is_absolute():
                p = (project_root / p).resolve()
            if p.exists() and p.resolve() not in {x.resolve() for x in config_paths}:
                config_paths.append(p.resolve())
        for p in config_paths:
            _update_config(p, engine_path, max_det=max_det, retina_masks=retina_masks, half=half, copy_to_project=True)
    else:
        print("[INFO] configs were not modified. Re-run with --update-configs to switch JSON model paths to the engine.")
        print(f"[INFO] engine={engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
