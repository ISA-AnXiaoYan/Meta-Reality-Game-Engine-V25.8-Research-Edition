#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_hit_protocol_event_point.py

修复：
    ImportError: cannot import name 'EVENT_POINT' from 'hit_protocol'

根因：
    bullet_event_sender.py 已经是支持实时点流的新版本，
    但 hit_protocol.py 还是旧版本，缺少 EVENT_POINT / REASON_VISUAL_POINT，
    也可能缺少 BulletEvent 实时点字段。

用法：
    cd ~/PycharmProjects/Ids_Test_3.9/test607
    python3 tools/patch_hit_protocol_event_point.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path.cwd().resolve()
TARGETS = [
    ROOT / "hit_protocol.py",
    ROOT / "rentijiance" / "hit_protocol.py",
    ROOT / "cpp_bullet_core" / "hit_protocol.py",
]


EVENT_POINT_LINE = 'EVENT_POINT     = "point"      # 实时轨迹点采样：用于 UE5 实时可视化，也可驱动 4ms 级 segment 命中判定\n'
REASON_VISUAL_POINT_LINE = 'REASON_VISUAL_POINT    = "visual_point"  # 实时点流原因：每个新轨迹点直接推送给 UE5\n'

POINT_FIELDS_BLOCK = """
    # ---- 实时点流字段：事件相机坐标系，供 hit_judge 投影成 MVS 坐标后推送 UE ----
    # point_index 是同一 bullet_id 内的实时点序号，只用于 overlay_bullet_point。
    point_index: int = -1
    point_status: str = ""  # tracking / ended / lost；实时点默认 tracking

    # ---- 可视化轨迹点：事件相机坐标系，供 hit_judge 投影成 MVS 坐标后推送 UE ----
    # 格式: [[timestamp_us, x, y], ...]，按时间升序。为空时服务端退化为 approach/event 两点。
    track_points: List[List[float]] = field(default_factory=list)

    # ---- 连续轨迹质量字段：用于过滤人身上冒出来的伪 bullet ----
    track_point_count: int = 0
    track_path_len_px: float = 0.0
    track_duration_ms: float = 0.0
    track_displacement_px: float = 0.0
    track_straightness: float = 0.0
    track_line_error_px: float = 999.0
"""


def patch_one(path: Path) -> bool:
    if not path.exists():
        return False

    text = path.read_text(encoding="utf-8")
    original = text

    # 1) constants
    if "EVENT_POINT" not in text:
        anchor = 'EVENT_TERMINATE = "terminate"'
        idx = text.find(anchor)
        if idx >= 0:
            line_end = text.find("\n", idx)
            text = text[:line_end + 1] + EVENT_POINT_LINE + text[line_end + 1:]
        else:
            anchor2 = "REASON_UNKNOWN"
            idx2 = text.find(anchor2)
            if idx2 >= 0:
                line_start = text.rfind("\n", 0, idx2) + 1
                text = text[:line_start] + EVENT_POINT_LINE + text[line_start:]
            else:
                text = EVENT_POINT_LINE + text

    if "REASON_VISUAL_POINT" not in text:
        anchor = "REASON_UNKNOWN"
        idx = text.find(anchor)
        if idx >= 0:
            line_end = text.find("\n", idx)
            text = text[:line_end + 1] + REASON_VISUAL_POINT_LINE + text[line_end + 1:]
        else:
            anchor2 = "# 命中匹配方法"
            idx2 = text.find(anchor2)
            if idx2 >= 0:
                text = text[:idx2] + REASON_VISUAL_POINT_LINE + text[idx2:]
            else:
                text = text + "\n" + REASON_VISUAL_POINT_LINE

    # 2) BulletEvent fields
    missing_any_field = any(name not in text for name in [
        "point_index:",
        "point_status:",
        "track_points:",
        "track_point_count:",
        "track_line_error_px:",
    ])

    if missing_any_field:
        anchor = "    confidence: float = 1.0"
        idx = text.find(anchor)
        if idx >= 0:
            line_end = text.find("\n", idx)
            text = text[:line_end + 1] + POINT_FIELDS_BLOCK + text[line_end + 1:]
        else:
            anchor2 = "    terminate_reason:"
            idx2 = text.find(anchor2)
            if idx2 >= 0:
                line_end = text.find("\n", idx2)
                text = text[:line_end + 1] + POINT_FIELDS_BLOCK + text[line_end + 1:]
            else:
                raise SystemExit(
                    f"[ERROR] Cannot find insertion point for BulletEvent fields in {path}"
                )

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak_event_point")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"[OK] backup -> {backup}")
        path.write_text(text, encoding="utf-8")
        print(f"[OK] patched -> {path}")
    else:
        print(f"[OK] already compatible -> {path}")

    return True


def quick_import_test() -> None:
    print("\n[check] import hit_protocol constants")
    import importlib.util
    hp = ROOT / "hit_protocol.py"
    spec = importlib.util.spec_from_file_location("hit_protocol_test", str(hp))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    print("[OK] EVENT_POINT =", getattr(mod, "EVENT_POINT", None))
    print("[OK] REASON_VISUAL_POINT =", getattr(mod, "REASON_VISUAL_POINT", None))
    fields = getattr(mod.BulletEvent, "__dataclass_fields__", {})
    for name in ["point_index", "point_status", "track_points", "track_point_count", "track_line_error_px"]:
        print(f"[OK] BulletEvent.{name} =", "YES" if name in fields else "NO")


def main() -> int:
    print(f"[INFO] project root = {ROOT}")
    found = False
    for p in TARGETS:
        if patch_one(p):
            found = True
    if not found:
        raise SystemExit("[ERROR] no hit_protocol.py found")

    quick_import_test()
    print("\n[DONE] Now restart launch_fusion_system.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
