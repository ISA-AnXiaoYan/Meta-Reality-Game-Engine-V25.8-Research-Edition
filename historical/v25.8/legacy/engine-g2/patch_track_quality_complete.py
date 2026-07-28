from pathlib import Path
import re
import json
import shutil

def backup_once(p: Path, suffix: str):
    bak = p.with_suffix(p.suffix + suffix)
    if not bak.exists():
        shutil.copy2(p, bak)
    return bak

# ---------------------------------------------------------------------
# 1. patch hit_protocol.py: BulletEvent / HitCandidate 增加 track 字段
# ---------------------------------------------------------------------

def patch_protocol(path: str):
    p = Path(path)
    if not p.exists():
        print("skip missing:", p)
        return

    s = p.read_text(encoding="utf-8")
    bak = backup_once(p, ".bak_track_quality_protocol")

    if "track_point_count" not in s:
        s = s.replace(
            "    confidence: float = 1.0  # 0.0~1.0，seq_id 跳号时由服务端降低\n",
            "    confidence: float = 1.0  # 0.0~1.0，seq_id 跳号时由服务端降低\n\n"
            "    # ---- 连续轨迹质量字段：用于过滤人身上冒出来的伪 bullet ----\n"
            "    track_point_count: int = 0\n"
            "    track_path_len_px: float = 0.0\n"
            "    track_duration_ms: float = 0.0\n"
            "    track_displacement_px: float = 0.0\n"
            "    track_straightness: float = 0.0\n"
            "    track_line_error_px: float = 999.0\n",
            1,
        )

    if "track_point_count: int = 0" not in s[s.find("class HitCandidate"):]:
        s = s.replace(
            "    match_method:    str   = MATCH_POINT_ONLY\n",
            "    match_method:    str   = MATCH_POINT_ONLY\n\n"
            "    # ---- 连续轨迹质量字段，透传到 hit_candidate JSONL ----\n"
            "    track_point_count: int = 0\n"
            "    track_path_len_px: float = 0.0\n"
            "    track_duration_ms: float = 0.0\n"
            "    track_displacement_px: float = 0.0\n"
            "    track_straightness: float = 0.0\n"
            "    track_line_error_px: float = 999.0\n",
            1,
        )

    p.write_text(s, encoding="utf-8")
    print("patched protocol:", p, "backup:", bak)

# ---------------------------------------------------------------------
# 2. patch bullet_event_sender.py: 计算并发送轨迹质量
# ---------------------------------------------------------------------

TRACK_QUALITY_METHOD = r'''    def _track_quality(self, bullet_id: int) -> Dict[str, float]:
        """
        计算一个 bullet_id 的连续轨迹质量。

        这些字段只在 turn/terminate 事件里随 UDP 发出，不额外增加发送频率。
        用于 hit_judge 过滤人体边缘/衣服/反光产生的伪 bullet。
        """
        hist = self._point_history.get(bullet_id, [])
        n = len(hist)

        if n < 2:
            return {
                "track_point_count": int(n),
                "track_path_len_px": 0.0,
                "track_duration_ms": 0.0,
                "track_displacement_px": 0.0,
                "track_straightness": 0.0,
                "track_line_error_px": 999.0,
            }

        pts = [(float(x), float(y), int(t)) for x, y, t in hist]

        path_len = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            path_len += (dx * dx + dy * dy) ** 0.5

        x0, y0, t0 = pts[0]
        x1, y1, t1 = pts[-1]
        dx = x1 - x0
        dy = y1 - y0
        displacement = (dx * dx + dy * dy) ** 0.5
        duration_ms = max(0.0, (t1 - t0) / 1000.0)
        straightness = float(displacement / path_len) if path_len > 1e-6 else 0.0

        # 到首尾直线的 RMS 垂直距离。越小越像直线。
        if displacement < 1e-6:
            line_error = 999.0
        else:
            err2 = 0.0
            for x, y, _t in pts:
                # 2D 点到直线距离
                dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / displacement
                err2 += dist * dist
            line_error = (err2 / max(1, len(pts))) ** 0.5

        return {
            "track_point_count": int(n),
            "track_path_len_px": float(path_len),
            "track_duration_ms": float(duration_ms),
            "track_displacement_px": float(displacement),
            "track_straightness": float(straightness),
            "track_line_error_px": float(line_error),
        }

'''

NEW_FIND_APPROACH = r'''    def _find_approach_point(
        self, bullet_id: int, event_x: float, event_y: float
    ) -> Tuple[float, float, float, bool]:
        """
        从轨迹历史中找更早、更远的 approach_point。

        目的：
          用 approach_point -> event_point 这条较长轨迹段判断是否穿过人体 mask。
          不额外发送 path_sample，不增加 UDP 频率。
        """
        hist = self._point_history.get(bullet_id, [])
        if not hist:
            return event_x, event_y, 0.0, False

        # 至少拉出一段有意义的线段。太短的线段容易把人体局部噪声判成 HIT。
        min_segment_len = max(45.0, 8.0 * float(self.min_step_px))

        candidates = []
        for cx, cy, ts in hist:
            dx = float(cx) - float(event_x)
            dy = float(cy) - float(event_y)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= min_segment_len:
                candidates.append((float(cx), float(cy), int(ts), float(dist)))

        if candidates:
            # 取最早的候选点，覆盖更完整的飞行轨迹。
            cx, cy, _ts, dist = candidates[0]
            return cx, cy, dist, True

        # 兜底：保留旧逻辑，避免完全没有 approach。
        threshold = 2.0 * float(self.min_step_px)
        for cx, cy, _ts in reversed(hist):
            dx = float(cx) - float(event_x)
            dy = float(cy) - float(event_y)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= threshold:
                return float(cx), float(cy), float(dist), True

        return event_x, event_y, 0.0, False

'''

def patch_sender(path: str):
    p = Path(path)
    if not p.exists():
        print("skip missing:", p)
        return

    s = p.read_text(encoding="utf-8")
    bak = backup_once(p, ".bak_track_quality_sender")

    s = re.sub(
        r"POINT_HISTORY_MAXLEN\s*=\s*\d+(\s*#.*)?",
        "POINT_HISTORY_MAXLEN = 64  # 保留更多点，用于连续直线轨迹质量判断",
        s,
        count=1,
    )

    if "def _track_quality" not in s:
        s = s.replace(
            "    def _find_approach_point(\n",
            TRACK_QUALITY_METHOD + "\n    def _find_approach_point(\n",
            1,
        )

    pattern = re.compile(
        r"    def _find_approach_point\(\n"
        r"        self, bullet_id: int, event_x: float, event_y: float\n"
        r"    \) -> Tuple\[float, float, float, bool\]:\n"
        r".*?"
        r"(?=\n    def _estimate_speed)",
        re.S,
    )
    s, n = pattern.subn(NEW_FIND_APPROACH.rstrip() + "\n", s, count=1)
    if n != 1:
        raise SystemExit(f"failed to replace _find_approach_point in {p}")

    # _emit_turn 里加入 track_quality
    s = s.replace(
        "        approach_x, approach_y, approach_len, approach_valid = \\\n"
        "            self._find_approach_point(bullet_id, cx, cy)\n\n"
        "        event = BulletEvent(\n"
        "            camera_sn            = self.camera_sn,\n",
        "        approach_x, approach_y, approach_len, approach_valid = \\\n"
        "            self._find_approach_point(bullet_id, cx, cy)\n"
        "        track_quality = self._track_quality(bullet_id)\n\n"
        "        event = BulletEvent(\n"
        "            camera_sn            = self.camera_sn,\n",
        1,
    )

    # _emit_terminate 里加入 track_quality，replace 第二次同样片段
    s = s.replace(
        "        approach_x, approach_y, approach_len, approach_valid = \\\n"
        "            self._find_approach_point(bullet_id, cx, cy)\n\n"
        "        event = BulletEvent(\n"
        "            camera_sn            = self.camera_sn,\n",
        "        approach_x, approach_y, approach_len, approach_valid = \\\n"
        "            self._find_approach_point(bullet_id, cx, cy)\n"
        "        track_quality = self._track_quality(bullet_id)\n\n"
        "        event = BulletEvent(\n"
        "            camera_sn            = self.camera_sn,\n",
        1,
    )

    # 在 BulletEvent 构造里透传 **track_quality，两处 event 都会匹配
    if "**track_quality" not in s:
        s = s.replace(
            "            confidence           = 1.0,\n"
            "        )\n",
            "            confidence           = 1.0,\n"
            "            **track_quality,\n"
            "        )\n",
            2,
        )

    p.write_text(s, encoding="utf-8")
    print("patched sender:", p, "backup:", bak)

# ---------------------------------------------------------------------
# 3. patch hit_judge_server.py: 强制轨迹质量 + mask-only HIT
# ---------------------------------------------------------------------

TRACK_CONFIG_BLOCK = r'''        self.trajectory_segment_enable = _as_bool(self.hit_cfg.get("trajectory_segment_enable", True), True)
        self.trajectory_segment_lookback_ms = float(self.hit_cfg.get("trajectory_segment_lookback_ms", 12.0) or 12.0)

        # 连续直线轨迹质量门控：过滤人身上冒出来的伪 bullet
        self.track_quality_enable = _as_bool(self.hit_cfg.get("track_quality_enable", True), True)
        self.min_track_point_count = int(self.hit_cfg.get("min_track_point_count", 5) or 5)
        self.min_track_path_len_px = float(self.hit_cfg.get("min_track_path_len_px", 80.0) or 80.0)
        self.min_track_duration_ms = float(self.hit_cfg.get("min_track_duration_ms", 15.0) or 15.0)
        self.min_track_straightness = float(self.hit_cfg.get("min_track_straightness", 0.85) or 0.85)
        self.max_track_line_error_px = float(self.hit_cfg.get("max_track_line_error_px", 12.0) or 12.0)
        self.min_hit_avg_speed_px_per_ms = float(self.hit_cfg.get("min_hit_avg_speed_px_per_ms", 0.6) or 0.6)
        self.min_hit_approach_len_px = float(self.hit_cfg.get("min_hit_approach_len_px", 45.0) or 45.0)
        self.reject_terminate_reasons_for_hit = set(self.hit_cfg.get("reject_terminate_reasons_for_hit", ["static", "probation", "probation_offset"]) or ["static", "probation", "probation_offset"])

        # 最终 HIT 只认人体 mask，bbox 只作为 debug 参考
        self.final_hit_mask_only = _as_bool(self.hit_cfg.get("final_hit_mask_only", True), True)
        self.allow_bbox_final_hit = _as_bool(self.hit_cfg.get("allow_bbox_final_hit", False), False)
'''

TRACK_GATE_BLOCK = r'''        # 轨迹质量统计，来自 BulletEvent。缺失时默认为 0，并直接被 no_valid_track 拦截。
        track_point_count = int(getattr(ev, "track_point_count", 0) or 0)
        track_path_len_px = float(getattr(ev, "track_path_len_px", 0.0) or 0.0)
        track_duration_ms = float(getattr(ev, "track_duration_ms", 0.0) or 0.0)
        track_displacement_px = float(getattr(ev, "track_displacement_px", 0.0) or 0.0)
        track_straightness = float(getattr(ev, "track_straightness", 0.0) or 0.0)
        track_line_error_px = float(getattr(ev, "track_line_error_px", 999.0) or 999.0)
        approach_len_px = float(getattr(ev, "approach_len_px", 0.0) or 0.0)
        avg_speed_for_hit = float(getattr(ev, "avg_speed_px_per_ms", 0.0) or 0.0)

        geometry_summary.update({
            "track_point_count": track_point_count,
            "track_path_len_px": round(track_path_len_px, 3),
            "track_duration_ms": round(track_duration_ms, 3),
            "track_displacement_px": round(track_displacement_px, 3),
            "track_straightness": round(track_straightness, 3),
            "track_line_error_px": round(track_line_error_px, 3),
            "approach_len_px": round(approach_len_px, 3),
            "avg_speed_px_per_ms": round(avg_speed_for_hit, 3),
            "track_quality_enable": bool(self.track_quality_enable),
            "final_hit_mask_only": bool(self.final_hit_mask_only),
            "allow_bbox_final_hit": bool(self.allow_bbox_final_hit),
        })

        if self.track_quality_enable:
            reject_reason = None

            if str(getattr(ev, "terminate_reason", "")) in self.reject_terminate_reasons_for_hit:
                reject_reason = "rejected_terminate_reason"
            elif track_point_count < int(self.min_track_point_count):
                reject_reason = "track_too_few_points"
            elif track_path_len_px < float(self.min_track_path_len_px):
                reject_reason = "track_path_too_short"
            elif track_duration_ms < float(self.min_track_duration_ms):
                reject_reason = "track_duration_too_short"
            elif avg_speed_for_hit < float(self.min_hit_avg_speed_px_per_ms):
                reject_reason = "trajectory_too_slow"
            elif approach_len_px < float(self.min_hit_approach_len_px):
                reject_reason = "approach_segment_too_short"
            elif track_straightness < float(self.min_track_straightness):
                reject_reason = "track_not_straight"
            elif track_line_error_px > float(self.max_track_line_error_px):
                reject_reason = "track_line_error_too_large"

            if reject_reason:
                self._emit_debug(
                    cam,
                    ev,
                    "no_valid_track",
                    reject_reason,
                    period=period,
                    selection=selection_info,
                    geometry=geometry_summary,
                    age_ms=age_ms,
                )
                return "no_candidate"

'''

def patch_judge(path: str):
    p = Path(path)
    if not p.exists():
        print("skip missing:", p)
        return

    s = p.read_text(encoding="utf-8")
    bak = backup_once(p, ".bak_track_quality_judge")

    if "self.track_quality_enable" not in s:
        s = s.replace(
            "        self.trajectory_segment_enable = _as_bool(self.hit_cfg.get(\"trajectory_segment_enable\", True), True)\n"
            "        self.trajectory_segment_lookback_ms = float(self.hit_cfg.get(\"trajectory_segment_lookback_ms\", 12.0) or 12.0)\n",
            TRACK_CONFIG_BLOCK,
            1,
        )

    # debug base 里加入 track 字段
    if '"track_point_count": int(getattr(ev, "track_point_count", 0) or 0),' not in s:
        s = s.replace(
            '            "avg_speed_px_per_ms": float(ev.avg_speed_px_per_ms),\n',
            '            "avg_speed_px_per_ms": float(ev.avg_speed_px_per_ms),\n'
            '            "track_point_count": int(getattr(ev, "track_point_count", 0) or 0),\n'
            '            "track_path_len_px": round(float(getattr(ev, "track_path_len_px", 0.0) or 0.0), 3),\n'
            '            "track_duration_ms": round(float(getattr(ev, "track_duration_ms", 0.0) or 0.0), 3),\n'
            '            "track_displacement_px": round(float(getattr(ev, "track_displacement_px", 0.0) or 0.0), 3),\n'
            '            "track_straightness": round(float(getattr(ev, "track_straightness", 0.0) or 0.0), 3),\n'
            '            "track_line_error_px": round(float(getattr(ev, "track_line_error_px", 999.0) or 999.0), 3),\n',
            1,
        )

    # geometry_summary 后加入 track gate
    if "track_too_few_points" not in s:
        s = s.replace(
            '            "evidence_level": "none",\n'
            "        }\n"
            "        for entry in entries:\n",
            '            "evidence_level": "none",\n'
            "        }\n"
            + TRACK_GATE_BLOCK +
            "        for entry in entries:\n",
            1,
        )

    # bbox 不再作为最终 HIT：只允许 point_in_mask / segment_intersect_mask
    if "bbox_method_ignored_for_final_hit" not in s:
        s = s.replace(
            '                if space > float(geometry_summary.get("geometry_best_score", 0.0)):\n'
            '                    geometry_summary["geometry_best_method"] = str(method)\n'
            '                    geometry_summary["geometry_best_score"] = round(float(space), 3)\n'
            '                    geometry_summary["evidence_level"] = evidence_level\n'
            '                if period is not None:\n',
            '                if space > float(geometry_summary.get("geometry_best_score", 0.0)):\n'
            '                    geometry_summary["geometry_best_method"] = str(method)\n'
            '                    geometry_summary["geometry_best_score"] = round(float(space), 3)\n'
            '                    geometry_summary["evidence_level"] = evidence_level\n'
            '                if bool(self.final_hit_mask_only) and not bool(self.allow_bbox_final_hit):\n'
            '                    if str(method) not in {"point_in_mask", "segment_intersect_mask"}:\n'
            '                        geometry_summary["bbox_method_ignored_for_final_hit"] = str(method)\n'
            '                        continue\n'
            '                if period is not None:\n',
            1,
        )

    # HitCandidate 构造中写入 track 字段
    if "track_point_count=track_point_count" not in s:
        s = s.replace(
            "                    approach_valid=bool(use_segment),\n"
            "                    match_method=str(method),\n",
            "                    approach_valid=bool(use_segment),\n"
            "                    match_method=str(method),\n"
            "                    track_point_count=int(track_point_count),\n"
            "                    track_path_len_px=float(track_path_len_px),\n"
            "                    track_duration_ms=float(track_duration_ms),\n"
            "                    track_displacement_px=float(track_displacement_px),\n"
            "                    track_straightness=float(track_straightness),\n"
            "                    track_line_error_px=float(track_line_error_px),\n",
            1,
        )

    # candidate JSON 显式写入 track 配置和字段
    if 'd["track_quality_enable"] = bool(self.track_quality_enable)' not in s:
        s = s.replace(
            '        d["geometry_mode"] = best_meta.get("geometry_mode", best.match_method)\n',
            '        d["geometry_mode"] = best_meta.get("geometry_mode", best.match_method)\n'
            '        d["track_quality_enable"] = bool(self.track_quality_enable)\n'
            '        d["final_hit_mask_only"] = bool(self.final_hit_mask_only)\n'
            '        d["track_point_count"] = int(track_point_count)\n'
            '        d["track_path_len_px"] = round(float(track_path_len_px), 3)\n'
            '        d["track_duration_ms"] = round(float(track_duration_ms), 3)\n'
            '        d["track_displacement_px"] = round(float(track_displacement_px), 3)\n'
            '        d["track_straightness"] = round(float(track_straightness), 3)\n'
            '        d["track_line_error_px"] = round(float(track_line_error_px), 3)\n',
            1,
        )

    p.write_text(s, encoding="utf-8")
    print("patched judge:", p, "backup:", bak)

# ---------------------------------------------------------------------
# 4. 写入配置
# ---------------------------------------------------------------------

def patch_config(path: str):
    p = Path(path)
    if not p.exists():
        print("skip missing config:", p)
        return

    cfg = json.loads(p.read_text(encoding="utf-8"))
    bak = backup_once(p, ".bak_track_quality_config")

    hit = cfg.setdefault("hit_judge", {})
    hit["trajectory_segment_enable"] = True
    hit["track_quality_enable"] = True
    hit["min_track_point_count"] = 5
    hit["min_track_path_len_px"] = 80.0
    hit["min_track_duration_ms"] = 15.0
    hit["min_track_straightness"] = 0.85
    hit["max_track_line_error_px"] = 12.0
    hit["min_hit_avg_speed_px_per_ms"] = 0.6
    hit["min_hit_approach_len_px"] = 45.0
    hit["reject_terminate_reasons_for_hit"] = ["static", "probation", "probation_offset"]
    hit["final_hit_mask_only"] = True
    hit["allow_bbox_final_hit"] = False

    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("patched config:", p, "backup:", bak)

for f in ["hit_protocol.py", "cpp_bullet_core/hit_protocol.py"]:
    patch_protocol(f)

for f in ["bullet_event_sender.py", "cpp_bullet_core/bullet_event_sender.py"]:
    patch_sender(f)

for f in ["hit_judge_server.py", "rentijiance/hit_judge_server.py"]:
    patch_judge(f)

patch_config("hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json")
