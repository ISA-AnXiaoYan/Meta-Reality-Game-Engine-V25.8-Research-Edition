"""
cpp_line_motion_filter.py — Final bridge for the handwritten C++ LinearMotionFilter full port.

This wrapper is intentionally conservative:
- detection / state update runs in the C++ core;
- accepted_clusters and debug_rows keep the Python dict/list contract used by the main script;
- detection / state update stays in the C++ core;
- this version replaces only the display layer with confirmed-only yellow bbox + ID + short smoothed trail, with hidden pre-confirmation history backfill.

Final replacement bridge for the handwritten C++ bullet trajectory state machine.
This file intentionally changes only visualization: it does not change C++ detection, bullet_id assignment, CSV logging, hit events, OBS/SRT, or synchronization.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Iterable, List
from pathlib import Path
import sys

# Ensure the compiled pybind11 extension can be found both when it is copied
# to test607/ and when it remains under test607/cpp_bullet_core/.
_THIS_DIR = Path(__file__).resolve().parent
_CPP_CORE_DIR = _THIS_DIR / "cpp_bullet_core"
for _p in (str(_THIS_DIR), str(_CPP_CORE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

try:
    import cv2  # noqa: F401  # used by the original renderer methods
except Exception:  # pragma: no cover
    cv2 = None

from line_motion_filter_backfill import (
    LinearMotionFilter as _PythonLinearMotionFilterRenderer,
    TrackConfig,
    AssociationConfig,
    DrawConfig,
)

try:
    from _cpp_bullet_full_port_stage06 import (
        TrackConfig as _CppTrackConfig,
        AssociationConfig as _CppAssociationConfig,
        DrawConfig as _CppDrawConfig,
        CppLineMotionFilter as _Core,
    )
except Exception as exc:  # fail loudly; no silent fallback
    raise RuntimeError(
        "最终版 C++ 模块 _cpp_bullet_full_port_stage06 导入失败。"
        "请先在项目根目录运行 ./BUILD_CPP_SHOT_BOUNCE_LINK.sh；不要静默回退 Python。"
    ) from exc


def _copy_dataclass_to_cpp(src: Any, dst: Any) -> Any:
    """Copy same-named dataclass fields into the C++ pybind config object."""
    if src is None:
        return dst
    for f in fields(src):
        if hasattr(dst, f.name):
            setattr(dst, f.name, getattr(src, f.name))
    return dst


def _clusters_np_to_plain_list(clusters_np: Any) -> List[dict]:
    """Safe Python-side conversion of Metavision/spatter structured clusters.

    The pybind module also exposes an object iterator, but this wrapper keeps numpy
    structured scalar handling in Python to avoid ABI-specific surprises. The core still
    receives a compact list and performs all state-machine work in C++.
    """
    if clusters_np is None:
        return []
    out: List[dict] = []
    for c in clusters_np:
        try:
            raw_id = int(c["id"])
            x = float(c["x"])
            y = float(c["y"])
            w = float(c["width"])
            h = float(c["height"])
        except Exception:
            # Accept dict-like rows too, which is useful for validators.
            raw_id = int(c.get("id", c.get("raw_id", -1)))
            x = float(c.get("x", 0.0))
            y = float(c.get("y", 0.0))
            w = float(c.get("width", 1.0))
            h = float(c.get("height", 1.0))
        out.append({
            "id": raw_id,
            "raw_id": raw_id,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "cx": x + 0.5 * w,
            "cy": y + 0.5 * h,
        })
    return out


class CppLinearMotionFilter:
    """Python-compatible facade for the final handwritten C++ filter."""

    def __init__(self, track: TrackConfig | None = None,
                 assoc: AssociationConfig | None = None,
                 draw: DrawConfig | None = None):
        self.track = track or TrackConfig()
        self.assoc = assoc or AssociationConfig()
        self.draw_cfg = draw or DrawConfig()
        # P13 需要重新编译过的 C++ 模块；旧 .so 没有这些字段，会导致改动完全不生效。
        if not hasattr(_CppTrackConfig(), "bullet_id_stitch_reverse_enabled"):
            raise RuntimeError(
                "当前 _cpp_bullet_full_port_stage06 还是旧版，缺少 bullet_id_stitch_reverse_enabled。"
                "请在项目根目录运行 ./BUILD_CPP_SHOT_BOUNCE_LINK.sh 后再启动。"
            )
        if not hasattr(_CppTrackConfig(), "bullet_id_bounce_link_enabled"):
            raise RuntimeError(
                "当前 _cpp_bullet_full_port_stage06 还是旧版，缺少 bullet_id_bounce_link_enabled。"
                "请在项目根目录运行 ./BUILD_CPP_SHOT_BOUNCE_LINK.sh 重新编译 cpp_bullet_core"
            )
        ct = _copy_dataclass_to_cpp(self.track, _CppTrackConfig())
        ca = _copy_dataclass_to_cpp(self.assoc, _CppAssociationConfig())
        cd = _copy_dataclass_to_cpp(self.draw_cfg, _CppDrawConfig())
        self._core = _Core(ct, ca, cd)
        # Renderer-only Python object. This preserves the exact original pixel-level
        # _draw_cyber_trail_roi implementation while C++ owns trajectory state.
        self._renderer = _PythonLinearMotionFilterRenderer(
            track=self.track, assoc=self.assoc, draw=self.draw_cfg
        )
        self._last_debug_rows: List[dict] = []
        self._last_accepted: List[dict] = []

    def update(self, clusters_np: Any, ts: int) -> List[dict]:
        # Fast path for Metavision/spatter numpy structured arrays: C++ reads the dtype fields
        # directly, avoiding a Python loop that would rebuild hundreds of cluster dicts.
        if isinstance(clusters_np, np.ndarray) and clusters_np.dtype.fields is not None:
            accepted = self._core.update_numpy(clusters_np, int(ts))
        else:
            # Fallback for validators and tests that pass list[dict].
            clusters = _clusters_np_to_plain_list(clusters_np)
            accepted = self._core.update(clusters, int(ts))
        # The pybind update already returns accepted rows from debug_rows; keep a local copy
        # so repeated draw/get calls do not touch core internals unnecessarily.
        self._last_accepted = [dict(x) for x in accepted]
        self._last_debug_rows = [dict(x) for x in self._core.get_last_debug_rows()]
        return list(self._last_accepted)

    def get_last_debug_rows(self) -> List[dict]:
        return list(self._last_debug_rows)

    def get_draw_paths(self, include_inactive: bool = False) -> List[dict]:
        return [dict(x) for x in self._core.get_draw_paths(bool(include_inactive))]

    def debug_state_counts(self) -> dict:
        return dict(self._core.debug_state_counts())

    def draw(self, img: np.ndarray, accepted_clusters: Iterable[dict],
             box_color=(0, 255, 255), text_color=(0, 255, 255),
             draw_debug_boxes: bool = False, show_head: bool = True) -> None:
        """Strict display-only renderer: confirmed bullet boxes only.

        This function intentionally does NOT draw:
        - raw SpatterTracker clusters;
        - raw/stable/probation candidate tracks;
        - hidden pre-confirmation cache;
        - short trails from unconfirmed candidates;
        - hold/miss/terminated residual trajectories.

        A cluster is displayed only when the core has assigned a non-negative
        display_bullet_id. This changes only the UI rendering, not detection,
        CSV logs, bullet IDs, hit events, or TSF1/OBS publishing.
        """
        import cv2 as _cv2

        YELLOW = (0, 255, 255)
        BLACK = (0, 0, 0)
        BBOX_MIN_SIZE = 10
        BBOX_THICKNESS = 2
        ID_FONT_SCALE = 0.60
        ID_THICKNESS = 2

        def _to_int(v, default=-1):
            try:
                return int(v)
            except Exception:
                return default

        def _to_float(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return default

        def _display_bullet_id(c: dict) -> int:
            # The only display gate: final committed bullet id.
            # display_bullet_id < 0 means raw/stable/probation candidate and must not be drawn.
            return _to_int(c.get("display_bullet_id", -1), -1)

        def _cluster_center(c: dict):
            if "cx" in c and "cy" in c:
                return _to_float(c.get("cx")), _to_float(c.get("cy"))
            x = _to_float(c.get("x", 0.0))
            y = _to_float(c.get("y", 0.0))
            w = _to_float(c.get("width", 1.0), 1.0)
            h = _to_float(c.get("height", 1.0), 1.0)
            return x + 0.5 * w, y + 0.5 * h

        def _bbox_from_cluster(c: dict, cx: float, cy: float):
            w = max(1.0, _to_float(c.get("width", BBOX_MIN_SIZE), BBOX_MIN_SIZE))
            h = max(1.0, _to_float(c.get("height", BBOX_MIN_SIZE), BBOX_MIN_SIZE))
            size = max(float(BBOX_MIN_SIZE), w, h)
            bx1 = int(round(cx - 0.5 * size))
            by1 = int(round(cy - 0.5 * size))
            bx2 = int(round(cx + 0.5 * size))
            by2 = int(round(cy + 0.5 * size))
            bx1 = max(0, min(img.shape[1] - 1, bx1))
            bx2 = max(0, min(img.shape[1] - 1, bx2))
            by1 = max(0, min(img.shape[0] - 1, by1))
            by2 = max(0, min(img.shape[0] - 1, by2))
            return bx1, by1, bx2, by2

        confirmed = []
        for c0 in (accepted_clusters or []):
            c = dict(c0)
            bid = _display_bullet_id(c)
            if bid < 0:
                continue
            cx, cy = _cluster_center(c)
            bx1, by1, bx2, by2 = _bbox_from_cluster(c, cx, cy)
            phase = str(c.get("phase", ""))
            label = f"b_{bid}"
            if phase == "maintain_turn":
                label += "*"
            elif phase.startswith("ghost_capture"):
                label += "~"
            confirmed.append((bid, bx1, by1, bx2, by2, label))

        # Do not draw anything if no final bullet exists in this slice.
        if not confirmed:
            return

        for bid, bx1, by1, bx2, by2, label in confirmed:
            _cv2.rectangle(img, (bx1, by1), (bx2, by2), BLACK, BBOX_THICKNESS + 2, _cv2.LINE_AA)
            _cv2.rectangle(img, (bx1, by1), (bx2, by2), YELLOW, BBOX_THICKNESS, _cv2.LINE_AA)

            tx = bx1
            ty = max(18, by1 - 6)
            if ty <= 18 and by2 + 18 < img.shape[0]:
                ty = by2 + 18
            tx = min(max(5, tx), max(5, img.shape[1] - 90))
            ty = min(max(18, ty), max(18, img.shape[0] - 6))
            _cv2.putText(img, label, (tx, ty),
                         _cv2.FONT_HERSHEY_SIMPLEX, ID_FONT_SCALE, BLACK,
                         ID_THICKNESS + 3, _cv2.LINE_AA)
            _cv2.putText(img, label, (tx, ty),
                         _cv2.FONT_HERSHEY_SIMPLEX, ID_FONT_SCALE, YELLOW,
                         ID_THICKNESS, _cv2.LINE_AA)


# Compatibility alias used by patched main scripts.
LinearMotionFilter = CppLinearMotionFilter
