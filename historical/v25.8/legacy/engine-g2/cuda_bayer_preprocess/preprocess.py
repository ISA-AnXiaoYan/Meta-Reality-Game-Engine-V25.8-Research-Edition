# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

_EXT = None
_PATTERN_TO_ID = {"RG": 0, "BG": 1, "GR": 2, "GB": 3}


def _norm_pattern(p: str) -> str:
    s = str(p or "BG").upper().replace("BAYER", "").replace("8", "")
    return s if s in _PATTERN_TO_ID else "BG"


def _device_string(device: str) -> str:
    d = str(device or "cuda:0")
    if d.isdigit():
        return "cuda:" + d
    if d == "cuda":
        return "cuda:0"
    return d


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    here = Path(__file__).resolve().parent
    sources = [str(here / "bayer_preprocess.cpp"), str(here / "bayer_preprocess_kernel.cu")]
    build_dir = here / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    extra_cflags = ["-O3"]
    _EXT = load(
        name="bayer_preprocess_cuda_ext",
        sources=sources,
        build_directory=str(build_dir),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=bool(int(os.environ.get("BAYER_PREPROCESS_VERBOSE", "0"))),
    )
    return _EXT


def preprocess_bayer_batch_gpu(raw_frames: List[Any], imgsz: int, bayer_pattern: str = "BG", device: str = "cuda:0", dtype: str = "fp16", pad_value: int = 114) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Convert same-size RawBayerFrame objects to YOLO-ready CUDA tensor.

    Output: tensor [B,3,imgsz,imgsz] on CUDA, RGB, normalized 0..1, float16/float32.
    """
    if not raw_frames:
        raise ValueError("raw_frames is empty")
    h = int(raw_frames[0].height)
    w = int(raw_frames[0].width)
    if h <= 0 or w <= 0:
        raw0 = np.asarray(raw_frames[0].raw)
        h, w = raw0.shape[:2]
    arrs = []
    metas: List[Dict[str, Any]] = []
    out = int(imgsz)
    scale = min(float(out) / float(w), float(out) / float(h))
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    pad_x = (out - resized_w) * 0.5
    pad_y = (out - resized_h) * 0.5
    for item in raw_frames:
        raw = np.asarray(item.raw, dtype=np.uint8)
        if raw.ndim != 2:
            raise ValueError(f"raw Bayer frame must be HxW, got {raw.shape}")
        if raw.shape[0] != h or raw.shape[1] != w:
            raise ValueError(f"all frames in one GPU batch must have same size; got {raw.shape}, expected {(h,w)}")
        arrs.append(raw)
        metas.append({
            "orig_w": int(w), "orig_h": int(h),
            "input_w": int(out), "input_h": int(out),
            "scale": float(scale), "pad_x": float(pad_x), "pad_y": float(pad_y),
            "resized_w": int(resized_w), "resized_h": int(resized_h),
            "bayer_pattern": _norm_pattern(getattr(item, "bayer_pattern", bayer_pattern)),
            "fid": int(getattr(item, "fid", 0) or 0),
            "mvs_frame_num": int(getattr(item, "mvs_frame_num", 0) or 0),
        })
    raw_np = np.stack(arrs, axis=0)
    raw_cpu = torch.from_numpy(raw_np)
    try:
        raw_cpu = raw_cpu.pin_memory()
    except Exception:
        pass
    dev = torch.device(_device_string(device))
    raw_gpu = raw_cpu.to(dev, non_blocking=True)
    ext = _load_ext()
    use_half = str(dtype or "fp16").lower() in {"fp16", "half", "float16"}
    pat = _PATTERN_TO_ID[_norm_pattern(bayer_pattern)]
    out_tensor = ext.bayer_preprocess(raw_gpu, int(out), int(out), int(pat), bool(use_half), float(pad_value) / 255.0)
    return out_tensor, metas
