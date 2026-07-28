from pathlib import Path
import re
import json
import shutil

ROOT = Path.cwd()


def backup_once(p: Path, tag: str) -> Path:
    bak = p.with_suffix(p.suffix + tag)
    if not bak.exists():
        bak.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    return bak


def patch_hit_protocol(path: str):
    p = Path(path)
    if not p.exists():
        print('skip missing', p)
        return
    s = p.read_text(encoding='utf-8')
    backup_once(p, '.bak_safe_hit_logic')

    # Make sure mask match constants exist. Existing hit_judge may use raw strings,
    # but having constants documents the allowed final modes.
    if 'MATCH_POINT_IN_MASK' not in s:
        s = s.replace(
            'MATCH_POINT_ONLY        = "point_only"         # approach_valid=False，退化为点判定\n',
            'MATCH_POINT_ONLY        = "point_only"         # approach_valid=False，退化为点判定\n'
            'MATCH_POINT_IN_MASK     = "point_in_mask"      # 事件点落入人体分割 mask\n'
            'MATCH_SEGMENT_IN_MASK   = "segment_intersect_mask"  # 轨迹段穿过人体分割 mask\n',
            1,
        )

    # Add continuous-track quality fields to BulletEvent. Backward compatible:
    # old packets have default values; new sender fills these fields.
    if 'track_point_count:' not in s:
        needle = '    avg_speed_px_per_ms: float = 0.0  # 最近几帧平均速度（像素/毫秒）\n    segment_index:       int   = 0    # 当前轨迹段索引（每次 maintain_turn 后 +1）\n'
        repl = needle + '\n' + \
            '    # ---- 连续轨迹质量字段：用于过滤人体自身冒出的伪子弹点 ----\n' + \
            '    track_point_count:    int   = 0    # 本 bullet 最近轨迹点数量\n' + \
            '    track_path_len_px:    float = 0.0  # 轨迹折线总长度\n' + \
            '    track_duration_ms:    float = 0.0  # 轨迹持续时间\n' + \
            '    track_displacement_px: float = 0.0 # 起点到终点位移\n' + \
            '    track_straightness:   float = 0.0  # 位移/折线长度，越接近1越直\n' + \
            '    track_line_error_px:  float = 9999.0 # 到首尾连线的 RMS 垂距\n'
        if needle not in s:
            raise SystemExit(f'cannot patch BulletEvent track fields in {p}')
        s = s.replace(needle, repl, 1)

    p.write_text(s, encoding='utf-8')
    print('patched hit_protocol:', p)


TRACK_METRICS_METHOD = r'''    def _track_metrics(self, bullet_id: int, event_x: float, event_y: float, event_ts: int) -> Dict[str, float]:
        """
        计算一个 bullet_id 的连续轨迹质量。只在 turn/terminate 发包时计算，
        不增加 UDP 发送频率，不影响 event_trigger CSV。
        """
        hist = list(self._point_history.get(bullet_id, []))
        if not hist:
            hist = []
        # 确保当前事件点参与统计。
        if not hist or abs(float(hist[-1][0]) - float(event_x)) > 0.1 or abs(float(hist[-1][1]) - float(event_y)) > 0.1:
            hist.append((float(event_x), float(event_y), int(event_ts)))

        n = len(hist)
        if n < 2:
            return {
                "track_point_count": n,
                "track_path_len_px": 0.0,
                "track_duration_ms": 0.0,
                "track_displacement_px": 0.0,
                "track_straightness": 0.0,
                "track_line_error_px": 9999.0,
            }

        path_len = 0.0
        for i in range(1, n):
            dx = float(hist[i][0]) - float(hist[i-1][0])
            dy = float(hist[i][1]) - float(hist[i-1][1])
            path_len += (dx * dx + dy * dy) ** 0.5

        x0, y0, t0 = hist[0]
        x1, y1, t1 = hist[-1]
        vx = float(x1) - float(x0)
        vy = float(y1) - float(y0)
        displacement = (vx * vx + vy * vy) ** 0.5
        duration_ms = max(0.0, (int(t1) - int(t0)) / 1000.0)
        straightness = displacement / max(1e-6, path_len)

        # RMS distance to the first-last line.
        denom = max(1e-6, displacement)
        err2 = 0.0
        for x, y, _ts in hist:
            # cross product magnitude / line length
            d = abs(vx * (float(y0) - float(y)) - (float(x0) - float(x)) * vy) / denom
            err2 += d * d
        line_error = (err2 / max(1, n)) ** 0.5

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
        从轨迹池中选择更早、更远的 approach 点，供 hit_judge 判断
        approach -> event 这条线段是否穿过人体 mask。

        注意：这里只改变随 turn/terminate 一起发送的 approach 点，
        不额外发送 path_sample，不会增加 UDP 频率。
        """
        hist = self._point_history.get(bullet_id, [])
        if not hist:
            return event_x, event_y, 0.0, False

        min_segment_len = max(45.0, 10.0 * float(self.min_step_px))
        candidates = []
        for cx, cy, _ts in hist:
            dx = float(cx) - float(event_x)
            dy = float(cy) - float(event_y)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= min_segment_len:
                candidates.append((float(cx), float(cy), float(dist)))

        if candidates:
            # 取最早的候选点，尽可能覆盖完整飞行路径。
            cx, cy, dist = candidates[0]
            return cx, cy, dist, True

        # 兜底：保留旧逻辑，找一个至少离当前点 2*min_step 的前序点。
        threshold = 2.0 * float(self.min_step_px)
        for cx, cy, _ts in reversed(hist):
            dx = float(cx) - float(event_x)
            dy = float(cy) - float(event_y)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= threshold:
                return float(cx), float(cy), float(dist), True

        return event_x, event_y, 0.0, False
'''


def patch_bullet_sender(path: str):
    p = Path(path)
    if not p.exists():
        print('skip missing', p)
        return
    s = p.read_text(encoding='utf-8')
    backup_once(p, '.bak_safe_hit_logic')

    # No path_sample allowed in this safe patch.
    if '_maybe_emit_path_sample' in s or 'SAMPLE_MIN_INTERVAL_US' in s:
        print(f'WARNING: {p} still contains path_sample code. Restore previous backup before applying this safe patch.')

    s = re.sub(
        r'POINT_HISTORY_MAXLEN\s*=\s*\d+(\s*#.*)?',
        'POINT_HISTORY_MAXLEN = 64  # 保留更多坐标历史，用于轨迹质量过滤和长线段判定',
        s,
        count=1,
    )

    # Replace _find_approach_point.
    pat = re.compile(
        r'    def _find_approach_point\(\n'
        r'        self, bullet_id: int, event_x: float, event_y: float\n'
        r'    \) -> Tuple\[float, float, float, bool\]:\n'
        r'.*?'
        r'(?=\n    def _estimate_speed)',
        re.S,
    )
    s2, n = pat.subn(NEW_FIND_APPROACH.rstrip() + '\n\n', s, count=1)
    if n != 1:
        raise SystemExit(f'cannot replace _find_approach_point in {p}')
    s = s2

    # Insert _track_metrics before _find_approach_point if absent.
    if 'def _track_metrics' not in s:
        s = s.replace('    def _find_approach_point(\n', TRACK_METRICS_METHOD + '\n    def _find_approach_point(\n', 1)

    # Add metrics before BulletEvent creation in emit_turn and emit_terminate.
    if 'metrics = self._track_metrics(bullet_id, cx, cy, timestamp_us)' not in s:
        s = s.replace(
            '        approach_x, approach_y, approach_len, approach_valid = \\\n            self._find_approach_point(bullet_id, cx, cy)\n\n        event = BulletEvent(',
            '        approach_x, approach_y, approach_len, approach_valid = \\\n            self._find_approach_point(bullet_id, cx, cy)\n        metrics = self._track_metrics(bullet_id, cx, cy, timestamp_us)\n\n        event = BulletEvent(',
            1,
        )
        s = s.replace(
            '        approach_x, approach_y, approach_len, approach_valid = \\\n            self._find_approach_point(bullet_id, cx, cy)\n\n        event = BulletEvent(',
            '        approach_x, approach_y, approach_len, approach_valid = \\\n            self._find_approach_point(bullet_id, cx, cy)\n        metrics = self._track_metrics(bullet_id, cx, cy, timestamp_us)\n\n        event = BulletEvent(',
            1,
        )

    # Add metric fields into BulletEvent constructor after avg_speed in both emit funcs.
    if 'track_point_count     = int(metrics.get("track_point_count", 0))' not in s:
        s = s.replace(
            '            avg_speed_px_per_ms  = speed,\n            segment_index        = seg_idx,',
            '            avg_speed_px_per_ms  = speed,\n'
            '            track_point_count    = int(metrics.get("track_point_count", 0)),\n'
            '            track_path_len_px    = float(metrics.get("track_path_len_px", 0.0)),\n'
            '            track_duration_ms    = float(metrics.get("track_duration_ms", 0.0)),\n'
            '            track_displacement_px = float(metrics.get("track_displacement_px", 0.0)),\n'
            '            track_straightness   = float(metrics.get("track_straightness", 0.0)),\n'
            '            track_line_error_px  = float(metrics.get("track_line_error_px", 9999.0)),\n'
            '            segment_index        = seg_idx,',
        )

    p.write_text(s, encoding='utf-8')
    print('patched bullet sender:', p)


TRACK_GATE_HELPER = r'''    def _track_quality_ok(self, ev: BulletEvent, geometry_summary: Dict[str, Any]) -> Tuple[bool, str]:
        """
        轨迹质量门控：过滤人体自身边缘/衣服/手臂产生的伪子弹点。
        只在 hit_judge 内判断，不改变事件相机发送频率。
        """
        try:
            terminate_reason = str(getattr(ev, "terminate_reason", "") or "")
            avg_speed = float(getattr(ev, "avg_speed_px_per_ms", 0.0) or 0.0)
            approach_len = float(getattr(ev, "approach_len_px", 0.0) or 0.0)
            track_points = int(getattr(ev, "track_point_count", 0) or 0)
            track_path = float(getattr(ev, "track_path_len_px", 0.0) or 0.0)
            track_duration = float(getattr(ev, "track_duration_ms", 0.0) or 0.0)
            straightness = float(getattr(ev, "track_straightness", 0.0) or 0.0)
            line_error = float(getattr(ev, "track_line_error_px", 9999.0) or 9999.0)
        except Exception:
            return False, "bad_track_fields"

        geometry_summary.update({
            "terminate_reason": terminate_reason,
            "avg_speed_px_per_ms": round(avg_speed, 3),
            "approach_len_px": round(approach_len, 3),
            "track_point_count": track_points,
            "track_path_len_px": round(track_path, 3),
            "track_duration_ms": round(track_duration, 3),
            "track_straightness": round(straightness, 3),
            "track_line_error_px": round(line_error, 3),
            "track_quality_enable": bool(self.track_quality_enable),
            "final_hit_mask_only": bool(self.final_hit_mask_only),
        })

        if not bool(self.track_quality_enable):
            return True, "disabled"

        if terminate_reason in self.reject_terminate_reasons_for_hit:
            return False, "rejected_terminate_reason"
        if not bool(getattr(ev, "approach_valid", False)):
            return False, "missing_approach"
        if approach_len < float(self.min_hit_approach_len_px):
            return False, "approach_too_short"
        if avg_speed < float(self.min_hit_avg_speed_px_per_ms):
            return False, "trajectory_too_slow"
        if track_points < int(self.min_track_point_count):
            return False, "track_too_few_points"
        if track_path < float(self.min_track_path_len_px):
            return False, "track_path_too_short"
        if track_duration < float(self.min_track_duration_ms):
            return False, "track_duration_too_short"
        if straightness < float(self.min_track_straightness):
            return False, "track_not_straight"
        if line_error > float(self.max_track_line_error_px):
            return False, "track_line_error_too_large"
        return True, "ok"
'''


def patch_hit_judge(path: str):
    p = Path(path)
    if not p.exists():
        print('skip missing', p)
        return
    s = p.read_text(encoding='utf-8')
    backup_once(p, '.bak_safe_hit_logic')

    # Add init config after trajectory_segment_lookback_ms line.
    if 'self.track_quality_enable' not in s:
        s = re.sub(
            r'(        self\.trajectory_segment_lookback_ms = .*?\n)',
            r'\1'
            r'        self.track_quality_enable = _as_bool(self.hit_cfg.get("track_quality_enable", True), True)\n'
            r'        self.final_hit_mask_only = _as_bool(self.hit_cfg.get("final_hit_mask_only", True), True)\n'
            r'        self.allow_bbox_final_hit = _as_bool(self.hit_cfg.get("allow_bbox_final_hit", False), False)\n'
            r'        self.min_hit_approach_len_px = float(self.hit_cfg.get("min_hit_approach_len_px", 45.0) or 45.0)\n'
            r'        self.min_hit_avg_speed_px_per_ms = float(self.hit_cfg.get("min_hit_avg_speed_px_per_ms", 0.6) or 0.6)\n'
            r'        self.min_track_point_count = int(self.hit_cfg.get("min_track_point_count", 5) or 5)\n'
            r'        self.min_track_path_len_px = float(self.hit_cfg.get("min_track_path_len_px", 80.0) or 80.0)\n'
            r'        self.min_track_duration_ms = float(self.hit_cfg.get("min_track_duration_ms", 15.0) or 15.0)\n'
            r'        self.min_track_straightness = float(self.hit_cfg.get("min_track_straightness", 0.85) or 0.85)\n'
            r'        self.max_track_line_error_px = float(self.hit_cfg.get("max_track_line_error_px", 12.0) or 12.0)\n'
            r'        self.reject_terminate_reasons_for_hit = set(self.hit_cfg.get("reject_terminate_reasons_for_hit", ["static", "probation", "probation_offset"]))\n',
            s,
            count=1,
        )

    # Add helper before judge_event.
    if 'def _track_quality_ok' not in s:
        s = s.replace('    def judge_event(self, ev: BulletEvent, cam: Optional[str] = None, age_ms: float = 0.0, force_final: bool = False) -> str:\n', TRACK_GATE_HELPER + '\n    def judge_event(self, ev: BulletEvent, cam: Optional[str] = None, age_ms: float = 0.0, force_final: bool = False) -> str:\n', 1)

    # Modify event_pt/use_segment/geometry_summary block and insert track gate.
    new = '''        event_pt = _apply_H(H, float(ev.event_x), float(ev.event_y))\n        # 只使用随 turn/terminate 一起携带的 approach->event 轨迹段，\n        # 不在事件端高频发送 path_sample。\n        try:\n            approach_len_px = float(getattr(ev, "approach_len_px", 0.0) or 0.0)\n        except Exception:\n            approach_len_px = 0.0\n        use_segment = bool(\n            self.trajectory_segment_enable\n            and bool(getattr(ev, "approach_valid", False))\n            and approach_len_px >= float(self.min_hit_approach_len_px)\n        )\n        approach_pt = _apply_H(H, float(ev.approach_x), float(ev.approach_y)) if use_segment else event_pt\n        candidates_meta: List[Tuple[HitCandidate, PersonEntry, Dict[str, Any]]] = []\n        geometry_summary: Dict[str, Any] = {\n            "event_point_mvs": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],\n            "approach_point_mvs": [round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],\n            "trajectory_segment_used": bool(use_segment),\n            "min_hit_approach_len_px": round(float(self.min_hit_approach_len_px), 3),\n            "geometry_tested_persons": 0,\n            "geometry_best_method": None,\n            "geometry_best_score": 0.0,\n            "geometry_best_bbox_method": None,\n            "geometry_best_bbox_score": 0.0,\n            "evidence_level": "none",\n        }\n        track_ok, track_reason = self._track_quality_ok(ev, geometry_summary)\n        if not track_ok:\n            self._emit_debug(cam, ev, "no_valid_track", track_reason, period=period, selection=selection_info, geometry=geometry_summary, age_ms=age_ms)\n            return "no_candidate"\n'''
    pat_geom = re.compile(
        r'        event_pt = _apply_H\(H, float\(ev\.event_x\), float\(ev\.event_y\)\)\n'
        r'.*?'
        r'        geometry_summary: Dict\[str, Any\] = \{\n'
        r'.*?'
        r'        \}\n'
        r'(?=        for entry in entries:)',
        re.S,
    )
    s2, n = pat_geom.subn(new, s, count=1)
    if n != 1 and 'track_ok, track_reason = self._track_quality_ok' not in s:
        raise SystemExit(f'cannot locate geometry init block in {p}')
    if n == 1:
        s = s2

    # Replace bbox fallback: keep bbox for debug only unless explicitly allowed.
    old2 = '''                if method is None:\n                    if _point_in_rect(event_pt, rect):\n                        method = MATCH_POINT_IN_BBOX\n                        space = 0.50\n                        evidence_level = "medium"\n                    elif use_segment and _segment_intersects_rect(approach_pt, event_pt, rect):\n                        method = MATCH_SEGMENT_INTERSECT\n                        space = 0.40\n                        evidence_level = "weak"\n                    else:\n                        d_rect = _dist_to_rect(event_pt, rect)\n                        if d_rect <= self.near_px:\n                            method = MATCH_POINT_NEAR_BBOX\n                            space = max(0.20, 0.34 * (1.0 - d_rect / max(1.0, self.near_px)))\n                            evidence_level = "weak"\n                if method is None:\n                    continue\n'''
    new2 = '''                bbox_method = None\n                bbox_space = 0.0\n                if _point_in_rect(event_pt, rect):\n                    bbox_method = MATCH_POINT_IN_BBOX\n                    bbox_space = 0.50\n                elif use_segment and _segment_intersects_rect(approach_pt, event_pt, rect):\n                    bbox_method = MATCH_SEGMENT_INTERSECT\n                    bbox_space = 0.40\n                else:\n                    d_rect = _dist_to_rect(event_pt, rect)\n                    if d_rect <= self.near_px:\n                        bbox_method = MATCH_POINT_NEAR_BBOX\n                        bbox_space = max(0.20, 0.34 * (1.0 - d_rect / max(1.0, self.near_px)))\n                if bbox_space > float(geometry_summary.get("geometry_best_bbox_score", 0.0)):\n                    geometry_summary["geometry_best_bbox_method"] = str(bbox_method) if bbox_method else None\n                    geometry_summary["geometry_best_bbox_score"] = round(float(bbox_space), 3)\n\n                # 最终 HIT 默认只允许人体 mask 命中。bbox 只做 debug，避免矩形框空白区域误判。\n                if method is None:\n                    if bool(self.allow_bbox_final_hit) and not bool(self.final_hit_mask_only) and bbox_method is not None:\n                        method = bbox_method\n                        space = bbox_space\n                        evidence_level = "weak"\n                    else:\n                        continue\n'''
    if old2 in s:
        s = s.replace(old2, new2, 1)
    elif 'geometry_best_bbox_method' not in s:
        raise SystemExit(f'cannot locate bbox fallback block in {p}')

    # Include track fields in output candidate JSON.
    if 'd["track_point_count"]' not in s:
        s = s.replace(
            '        d["geometry_mode"] = best_meta.get("geometry_mode", best.match_method)\n',
            '        d["geometry_mode"] = best_meta.get("geometry_mode", best.match_method)\n'
            '        d["final_hit_mask_only"] = bool(self.final_hit_mask_only)\n'
            '        d["allow_bbox_final_hit"] = bool(self.allow_bbox_final_hit)\n'
            '        d["track_quality_enable"] = bool(self.track_quality_enable)\n'
            '        d["track_point_count"] = int(getattr(ev, "track_point_count", 0) or 0)\n'
            '        d["track_path_len_px"] = round(float(getattr(ev, "track_path_len_px", 0.0) or 0.0), 3)\n'
            '        d["track_duration_ms"] = round(float(getattr(ev, "track_duration_ms", 0.0) or 0.0), 3)\n'
            '        d["track_straightness"] = round(float(getattr(ev, "track_straightness", 0.0) or 0.0), 3)\n'
            '        d["track_line_error_px"] = round(float(getattr(ev, "track_line_error_px", 9999.0) or 9999.0), 3)\n'
            '        d["approach_len_px"] = round(float(getattr(ev, "approach_len_px", 0.0) or 0.0), 3)\n',
            1,
        )

    p.write_text(s, encoding='utf-8')
    print('patched hit_judge:', p)


def patch_renderer(path: str):
    p = Path(path)
    if not p.exists():
        print('skip missing', p)
        return
    s = p.read_text(encoding='utf-8')
    backup_once(p, '.bak_safe_hit_logic')

    # Restrict displayed HIT modes to mask modes only when ok_modes set is present.
    s = re.sub(
        r'ok_modes\s*=\s*\{[^}]*\}',
        'ok_modes = {"point_in_mask", "segment_intersect_mask"}',
        s,
        count=1,
        flags=re.S,
    )
    # If no ok_modes exists, do nothing; hit_judge filtering is primary.
    p.write_text(s, encoding='utf-8')
    print('patched renderer:', p)


def patch_config(path: str):
    p = Path(path)
    if not p.exists():
        print('skip missing', p)
        return
    cfg = json.loads(p.read_text(encoding='utf-8'))
    backup_once(p, '.bak_safe_hit_logic')
    hit = cfg.setdefault('hit_judge', {})
    hit['trajectory_segment_enable'] = True
    hit['track_quality_enable'] = True
    hit['final_hit_mask_only'] = True
    hit['allow_bbox_final_hit'] = False
    hit['min_hit_approach_len_px'] = 45.0
    hit['min_hit_avg_speed_px_per_ms'] = 0.6
    hit['min_track_point_count'] = 5
    hit['min_track_path_len_px'] = 80.0
    hit['min_track_duration_ms'] = 15.0
    hit['min_track_straightness'] = 0.85
    hit['max_track_line_error_px'] = 12.0
    hit['reject_terminate_reasons_for_hit'] = ['static', 'probation', 'probation_offset']
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('patched config:', p)

for f in ['hit_protocol.py', 'cpp_bullet_core/hit_protocol.py']:
    patch_hit_protocol(f)
for f in ['bullet_event_sender.py', 'cpp_bullet_core/bullet_event_sender.py']:
    patch_bullet_sender(f)
for f in ['hit_judge_server.py', 'rentijiance/hit_judge_server.py']:
    patch_hit_judge(f)
for f in ['fusion_renderer_gl.py', 'rentijiance/fusion_renderer_gl.py']:
    patch_renderer(f)
patch_config('hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json')
print('safe hit logic patch complete')
