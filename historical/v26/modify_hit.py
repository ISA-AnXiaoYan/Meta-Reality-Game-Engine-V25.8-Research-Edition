from pathlib import Path
p=Path('/mnt/data/impact_hit_work/hit_judge_server.py')
s=p.read_text(encoding='utf-8')
s=s.replace('import argparse\n', 'import argparse\nimport bisect\n')
# replace TriggerCsvTailer class
start=s.index('class TriggerCsvTailer:')
end=s.index('\n\n@dataclass\nclass PersonEntry:', start)
new_trigger=r'''class TriggerCsvTailer:
    """Incrementally read event_trigger_{camera}.csv.

    It keeps both sync_index -> event_ts_us and a timestamp-sorted view so the
    hit judge can answer the important question:

        given a bullet impact timestamp, which 40 ms MVS trigger period did it
        happen in?

    This avoids using UDP receive time or YOLO write time for hit matching.
    """
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.fp = None
        self.offset = 0
        self.last_try = 0.0
        self.header: Optional[List[str]] = None
        self.map: Dict[int, int] = {}
        self._ts_items: List[Tuple[int, int]] = []  # (event_ts_us, sync_index)
        self._dirty = True
        self.last_error = "not opened"

    def poll(self) -> None:
        if self.fp is None:
            now = time.time()
            if now - self.last_try < 0.5:
                return
            self.last_try = now
            try:
                self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
                self.offset = 0
                self.header = None
                self.map.clear()
                self._dirty = True
            except Exception as exc:
                self.last_error = f"open failed: {exc}"
                return
        try:
            self.fp.seek(self.offset)
            while True:
                line = self.fp.readline()
                if not line:
                    break
                self.offset = self.fp.tell()
                line = line.strip()
                if not line:
                    continue
                parts = next(csv.reader([line]))
                if self.header is None:
                    self.header = parts
                    continue
                d = {self.header[i]: parts[i] for i in range(min(len(self.header), len(parts)))}
                try:
                    si = int(float(d.get("sync_index", "")))
                    ts = int(float(d.get("event_ts_us", "")))
                    # Some trigger CSVs contain helper rows with sync_index=-1.
                    # They are useful for diagnostics but not for MVS-frame anchoring.
                    if si >= 0 and ts > 0:
                        old = self.map.get(si)
                        self.map[si] = ts
                        if old != ts:
                            self._dirty = True
                except Exception:
                    continue
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"poll failed: {exc}"
            try:
                if self.fp is not None: self.fp.close()
            except Exception:
                pass
            self.fp = None

    def _ensure_sorted(self) -> None:
        if not self._dirty:
            return
        self._ts_items = sorted((int(ts), int(si)) for si, ts in self.map.items() if int(si) >= 0 and int(ts) > 0)
        self._dirty = False

    def find_period(self, event_ts_us: int, expected_period_us: int = 40000) -> Optional[Dict[str, Any]]:
        """Return the MVS trigger period containing event_ts_us.

        Returns None if the timestamp is earlier than the first known MVS trigger.
        If the next trigger is not in the CSV yet, it estimates the next boundary
        from expected_period_us.  The pending-event queue will still wait for the
        needed human_result frame before judging.
        """
        self._ensure_sorted()
        if not self._ts_items:
            return None
        ts_list = [x[0] for x in self._ts_items]
        i = bisect.bisect_right(ts_list, int(event_ts_us)) - 1
        if i < 0:
            return None
        anchor_ts, sync_index = self._ts_items[i]
        if i + 1 < len(self._ts_items):
            next_ts, next_sync = self._ts_items[i + 1]
            estimated_next = False
        else:
            next_ts = int(anchor_ts) + int(expected_period_us)
            next_sync = int(sync_index) + 1
            estimated_next = True
        period_us = max(1, int(next_ts) - int(anchor_ts))
        offset_us = int(event_ts_us) - int(anchor_ts)
        return {
            "sync_index": int(sync_index),
            "anchor_ts_us": int(anchor_ts),
            "next_sync_index": int(next_sync),
            "next_ts_us": int(next_ts),
            "period_us": int(period_us),
            "offset_us": int(offset_us),
            "offset_ms": float(offset_us) / 1000.0,
            "estimated_next": bool(estimated_next),
        }
'''
s=s[:start]+new_trigger+s[end:]
# add polygon helper after segment_intersects_rect
needle='''def _segment_intersects_rect(a: Tuple[float, float], b: Tuple[float, float], r: Tuple[float, float, float, float]) -> bool:\n    if _point_in_rect(a, r) or _point_in_rect(b, r):\n        return True\n    x1, y1, x2, y2 = r\n    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]\n    return any(_segments_intersect(a, b, c, d) for c, d in edges)\n'''
insert=r'''

def _point_in_polygon(p: Tuple[float, float], poly: Iterable[Iterable[float]]) -> bool:
    pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
    if len(pts) < 3:
        return False
    x, y = p
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)):
            denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
            x_cross = (xj - xi) * (y - yi) / denom + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _segment_intersects_polygon(a: Tuple[float, float], b: Tuple[float, float], poly: Iterable[Iterable[float]]) -> bool:
    pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
    if len(pts) < 3:
        return False
    if _point_in_polygon(a, pts) or _point_in_polygon(b, pts):
        return True
    for i in range(len(pts)):
        c = pts[i]
        d = pts[(i + 1) % len(pts)]
        if _segments_intersect(a, b, c, d):
            return True
    return False


def _person_mask_hit(person: Dict[str, Any], event_pt: Tuple[float, float], approach_pt: Tuple[float, float], approach_valid: bool) -> Tuple[Optional[str], float]:
    """Try precise segmentation-mask matching before bbox fallback."""
    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    if isinstance(polys, dict):
        polys = list(polys.values())
    if polys and isinstance(polys, list):
        # Human results usually store a list of polygons: [[[x,y],...], ...]
        for poly in polys:
            if not isinstance(poly, list):
                continue
            if approach_valid and _segment_intersects_polygon(approach_pt, event_pt, poly):
                return "segment_intersect_mask", 0.72
            if _point_in_polygon(event_pt, poly):
                return "point_in_mask", 0.66
    return None, 0.0
'''
s=s.replace(needle, needle+insert)
# Add Pending dataclass after PersonEntry
needle2='''@dataclass\nclass PersonEntry:\n    camera: str\n    camera_sn: str\n    event_serial: str\n    sync_index: int\n    event_ts_us: Optional[int]\n    frame_w: int\n    frame_h: int\n    persons: List[Dict[str, Any]]\n    raw: Dict[str, Any] = field(default_factory=dict)\n    wall_time: float = field(default_factory=time.time)\n'''
insert2=r'''


@dataclass
class PendingBulletEvent:
    ev: BulletEvent
    cam: str
    first_wall: float = field(default_factory=time.time)
    tries: int = 0
'''
s=s.replace(needle2, needle2+insert2)
# replace init counters block partially
old='''        self.seen_keys: Deque[Tuple[str, int, int]] = deque(maxlen=4096)
        self.seen_set = set()
        self.last_print = time.time()
        self.recv_count = 0
        self.cand_count = 0
        self.no_human_count = 0
        self.no_cam_count = 0
        self.max_dt_ms = float(self.hit_cfg.get("max_dt_ms", args.max_dt_ms))
        self.bbox_expand_px = float(self.hit_cfg.get("bbox_expand_px", args.bbox_expand_px))
        self.near_px = float(self.hit_cfg.get("near_bbox_px", args.near_bbox_px))
        self.min_score = float(self.hit_cfg.get("min_score_to_log", args.min_score_to_log))
        print(f"[hit_judge] config={cfg_path}", flush=True)
'''
new=r'''        self.seen_keys: Deque[Tuple[str, int, int]] = deque(maxlen=4096)
        self.seen_set = set()
        self.pending_events: Deque[PendingBulletEvent] = deque(maxlen=int(self.hit_cfg.get("pending_event_queue", 2048) or 2048))
        self.last_print = time.time()
        self.recv_count = 0
        self.cand_count = 0
        self.no_human_count = 0
        self.no_cam_count = 0
        self.nonimpact_count = 0
        self.wait_count = 0
        self.drop_count = 0
        self.max_dt_ms = float(self.hit_cfg.get("max_dt_ms", args.max_dt_ms))
        self.bbox_expand_px = float(self.hit_cfg.get("bbox_expand_px", args.bbox_expand_px))
        self.near_px = float(self.hit_cfg.get("near_bbox_px", args.near_bbox_px))
        self.min_score = float(self.hit_cfg.get("min_score_to_log", args.min_score_to_log))

        # Impact-timestamp-driven matching.  This is the more precise path:
        # bullet turn/terminate timestamp -> event_trigger period -> MVS human frame.
        self.impact_sync_enable = _as_bool(self.hit_cfg.get("impact_sync_enable", True), True)
        self.impact_event_types = set(self.hit_cfg.get("impact_event_types", [EVENT_TURN, EVENT_TERMINATE]) or [EVENT_TURN, EVENT_TERMINATE])
        self.impact_terminate_reasons = set(self.hit_cfg.get("impact_terminate_reasons", []) or [])
        self.impact_min_turn_angle_deg = float(self.hit_cfg.get("impact_min_turn_angle_deg", 0.0) or 0.0)
        self.impact_person_strategy = str(self.hit_cfg.get("impact_person_strategy", "nearest") or "nearest").lower()
        self.impact_pending_timeout_ms = float(self.hit_cfg.get("impact_pending_timeout_ms", 1500.0) or 1500.0)
        self.impact_max_offset_ms = float(self.hit_cfg.get("impact_max_offset_ms", 45.0) or 45.0)
        self.impact_use_mask = _as_bool(self.hit_cfg.get("impact_use_mask", True), True)
        sync_period_ms = float(self.sync_cfg.get("mvs_soft_trigger_period_ms", 40.0) or 40.0)
        self.expected_period_us = int(float(self.hit_cfg.get("impact_anchor_period_ms", sync_period_ms) or sync_period_ms) * 1000.0)
        self.impact_midpoint_ms = float(self.hit_cfg.get("impact_midpoint_ms", self.expected_period_us / 2000.0) or (self.expected_period_us / 2000.0))
        print(f"[hit_judge] config={cfg_path}", flush=True)
'''
s=s.replace(old,new)
# replace receive_udp through judge_event with new block
start=s.index('    def receive_udp(self) -> None:')
end=s.index('\n    def run(self) -> None:', start)
new_block=r'''    def receive_udp(self) -> None:
        while True:
            r, _, _ = select.select([self.sock], [], [], 0)
            if not r:
                return
            try:
                data, _addr = self.sock.recvfrom(65535)
                ev = BulletEvent.from_json(data)
            except Exception as exc:
                print(f"[hit_judge][WARN] bad udp packet: {exc}", flush=True)
                continue
            key = (str(ev.camera_sn), int(ev.bullet_id), int(ev.seq_id))
            if not self._mark_seen(key):
                continue
            self.recv_count += 1
            cam = self._camera_for_event(ev)
            if cam is None:
                self.no_cam_count += 1
                continue
            if self.impact_sync_enable and not self._is_impact_event(ev):
                self.nonimpact_count += 1
                continue
            self.pending_events.append(PendingBulletEvent(ev=ev, cam=cam))

    def _is_impact_event(self, ev: BulletEvent) -> bool:
        """Only turn/terminate-like events should trigger hit judgment.

        Normal bullet track points are visual data.  Hit judgment should be driven
        by physically meaningful moments: turn/bounce, disappearance, terminate.
        """
        if str(ev.event_type) not in self.impact_event_types:
            return False
        if ev.event_type == EVENT_TURN and float(ev.turn_angle_deg) < self.impact_min_turn_angle_deg:
            return False
        if ev.event_type == EVENT_TERMINATE and self.impact_terminate_reasons:
            return str(ev.terminate_reason) in self.impact_terminate_reasons
        return True

    def _person_entry_by_sync(self, cam: str, sync_index: int) -> Optional[PersonEntry]:
        for e in reversed(self.history.get(cam, [])):
            if int(e.sync_index) == int(sync_index):
                return e
        return None

    def _nearest_person_entry(self, cam: str, event_ts: int) -> Optional[PersonEntry]:
        # Kept as a fallback for impact_sync_enable=false.
        hist = [e for e in self.history.get(cam, []) if e.event_ts_us is not None]
        if not hist:
            return None
        best = min(hist, key=lambda e: abs(int(e.event_ts_us or 0) - int(event_ts)))
        if abs(int(best.event_ts_us or 0) - int(event_ts)) / 1000.0 > self.max_dt_ms:
            return None
        return best

    def _impact_period_for_event(self, cam: str, event_ts_us: int) -> Optional[Dict[str, Any]]:
        trig = self.trigger_tailers.get(cam)
        if trig is None:
            return None
        period = trig.find_period(int(event_ts_us), expected_period_us=self.expected_period_us)
        if period is None:
            return None
        # Reject clearly out-of-period events.  A small margin protects against
        # minor trigger jitter, not against assigning an event to the wrong MVS frame.
        if float(period.get("offset_ms", 0.0)) < -1.0:
            return None
        if float(period.get("offset_ms", 0.0)) > self.impact_max_offset_ms:
            return None
        return period

    def _entries_for_impact(self, cam: str, period: Dict[str, Any]) -> Tuple[List[PersonEntry], str, bool]:
        """Choose the human frame(s) for a bullet impact timestamp.

        Strategies:
          period  : always use MVS frame k where T_k <= impact < T_{k+1}
          nearest : use frame k for offset < midpoint, otherwise frame k+1
          union   : use both k and k+1 when available, otherwise wait/fallback
        Returns (entries, strategy_used, should_wait).
        """
        k = int(period["sync_index"])
        k_next = int(period.get("next_sync_index", k + 1))
        offset_ms = float(period.get("offset_ms", 0.0))
        curr = self._person_entry_by_sync(cam, k)
        nxt = self._person_entry_by_sync(cam, k_next)
        strategy = self.impact_person_strategy
        if strategy in {"period", "anchor", "current"}:
            if curr is None:
                return [], "period_wait", True
            return [curr], "period", False
        if strategy in {"union", "both"}:
            entries = [e for e in (curr, nxt) if e is not None]
            if not entries:
                return [], "union_wait", True
            # If the impact is in the second half, waiting briefly for the next
            # frame is usually worth it because the person's body may have moved.
            if offset_ms >= self.impact_midpoint_ms and nxt is None:
                return entries, "union_partial_wait_next", True
            return entries, "union", False
        # Default: nearest synced human frame.
        want_next = offset_ms >= self.impact_midpoint_ms
        target = nxt if want_next else curr
        if target is None:
            # Wait for the desired frame, but allow current as a late fallback
            # only after pending timeout in process_pending_events().
            return [], "nearest_wait_next" if want_next else "nearest_wait_current", True
        return [target], "nearest_next" if want_next else "nearest_current", False

    def process_pending_events(self) -> None:
        if not self.pending_events:
            return
        keep: Deque[PendingBulletEvent] = deque(maxlen=self.pending_events.maxlen or 2048)
        now = time.time()
        while self.pending_events:
            pe = self.pending_events.popleft()
            pe.tries += 1
            result = self.judge_event(pe.ev, pe.cam)
            if result == "wait":
                age_ms = (now - pe.first_wall) * 1000.0
                if age_ms <= self.impact_pending_timeout_ms:
                    keep.append(pe)
                    self.wait_count += 1
                else:
                    self.drop_count += 1
            elif result == "no_human":
                self.no_human_count += 1
            elif result == "no_cam":
                self.no_cam_count += 1
        self.pending_events = keep

    def judge_event(self, ev: BulletEvent, cam: Optional[str] = None) -> str:
        cam = cam or self._camera_for_event(ev)
        if cam is None:
            return "no_cam"
        H = self.H.get(cam)
        if H is None:
            return "drop"

        period: Optional[Dict[str, Any]] = None
        strategy_used = "nearest_time_fallback"
        if self.impact_sync_enable:
            period = self._impact_period_for_event(cam, int(ev.timestamp_us))
            if period is None:
                return "wait"
            entries, strategy_used, should_wait = self._entries_for_impact(cam, period)
            if should_wait and not entries:
                return "wait"
            if not entries:
                return "no_human"
        else:
            entry = self._nearest_person_entry(cam, int(ev.timestamp_us))
            if entry is None:
                return "no_human"
            entries = [entry]

        event_pt = _apply_H(H, float(ev.event_x), float(ev.event_y))
        approach_pt = _apply_H(H, float(ev.approach_x), float(ev.approach_y)) if ev.approach_valid else event_pt
        candidates: List[HitCandidate] = []
        for entry in entries:
            if entry is None or not entry.persons:
                continue
            for person in entry.persons:
                if not isinstance(person, dict):
                    continue
                box = person.get("bbox_xyxy") or person.get("bbox")
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                rect = _expand_bbox_xyxy(box, self.bbox_expand_px, entry.frame_w, entry.frame_h)
                method = None
                space = 0.0
                if self.impact_use_mask:
                    method, space = _person_mask_hit(person, event_pt, approach_pt, bool(ev.approach_valid))
                if method is None:
                    if ev.approach_valid and _segment_intersects_rect(approach_pt, event_pt, rect):
                        method = MATCH_SEGMENT_INTERSECT
                        space = 0.62
                    elif _point_in_rect(event_pt, rect):
                        method = MATCH_POINT_IN_BBOX
                        space = 0.50
                    else:
                        d = _dist_to_rect(event_pt, rect)
                        if d <= self.near_px:
                            method = MATCH_POINT_NEAR_BBOX
                            space = max(0.20, 0.42 * (1.0 - d / max(1.0, self.near_px)))
                if method is None:
                    continue
                if period is not None:
                    # This dt is not UDP/wall time.  It is the offset between the
                    # impact timestamp and the selected MVS frame's trigger time.
                    dt_ms = abs(int(entry.event_ts_us or 0) - int(ev.timestamp_us)) / 1000.0 if entry.event_ts_us else abs(float(period.get("offset_ms", 0.0)))
                    # Penalize within the 40 ms period gently.  The frame was selected
                    # by sync period, so large offsets are expected and not fatal.
                    time_score = max(0.0, 0.24 * (1.0 - min(dt_ms, self.expected_period_us / 1000.0) / max(1.0, self.expected_period_us / 1000.0)))
                else:
                    dt_ms = abs(int(entry.event_ts_us or 0) - int(ev.timestamp_us)) / 1000.0
                    time_score = max(0.0, 0.22 * (1.0 - dt_ms / max(1.0, self.max_dt_ms)))
                # Impact-like events are now the only events judged, so event score
                # reflects reliability: terminate/disappear/end > turn.
                event_score = 0.14 if ev.event_type == EVENT_TERMINATE else 0.11
                speed_score = min(0.08, max(0.0, float(ev.avg_speed_px_per_ms) / 3.0 * 0.08))
                score = ScoreBreakdown(time_score=time_score, space_score=space, event_score=event_score, speed_score=speed_score, loss_penalty=0.0)
                bbox = rect
                try:
                    raw_box = list(box)
                    if len(raw_box) >= 4:
                        x1, y1, x2, y2 = [float(v) for v in raw_box[:4]]
                        if x2 <= x1 or y2 <= y1:
                            x2, y2 = x1 + float(raw_box[2]), y1 + float(raw_box[3])
                        bbox = (int(round(x1)), int(round(y1)), int(round(x2-x1)), int(round(y2-y1)))
                except Exception:
                    bbox = (int(rect[0]), int(rect[1]), int(rect[2]-rect[0]), int(rect[3]-rect[1]))
                cand = HitCandidate(
                    camera_sn=str(ev.camera_sn),
                    camera_alias=cam,
                    bullet_id=int(ev.bullet_id),
                    seq_id=int(ev.seq_id),
                    stable_id=int(person.get("stable_id", person.get("person_id", 0)) or 0),
                    event_ts=int(ev.timestamp_us),
                    person_ts=int(entry.event_ts_us or 0),
                    dt_ms=float(dt_ms),
                    event_type=str(ev.event_type),
                    terminate_reason=str(ev.terminate_reason),
                    turn_angle_deg=float(ev.turn_angle_deg),
                    event_point=(float(event_pt[0]), float(event_pt[1])),
                    approach_point=(float(approach_pt[0]), float(approach_pt[1])),
                    approach_valid=bool(ev.approach_valid),
                    match_method=str(method),
                    person_bbox=tuple(int(v) for v in bbox),
                    person_centroid=tuple(person.get("foot_pixel") or person.get("centroid") or (0.0, 0.0)),
                    person_confirmed=True,
                    person_held=False,
                    score_breakdown=score,
                )
                if cand.total_score >= self.min_score:
                    candidates.append(cand)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        if not candidates:
            return "no_candidate"
        best = candidates[0]
        d = best.to_dict()
        d["source"] = "hit_judge_server"
        d["judge_mode"] = "impact_sync" if self.impact_sync_enable else "nearest_time_fallback"
        d["human_match_strategy"] = strategy_used
        if period is not None:
            d["impact_event_ts_us"] = int(ev.timestamp_us)
            d["impact_sync_index"] = int(period["sync_index"])
            d["impact_anchor_ts_us"] = int(period["anchor_ts_us"])
            d["impact_next_sync_index"] = int(period["next_sync_index"])
            d["impact_next_ts_us"] = int(period["next_ts_us"])
            d["impact_offset_ms"] = round(float(period["offset_ms"]), 3)
            d["impact_period_ms"] = round(float(period["period_us"]) / 1000.0, 3)
            d["impact_next_boundary_estimated"] = bool(period.get("estimated_next", False))
        d["mvs_sync_index"] = int(self._person_entry_by_sync(cam, int(best.person_ts)) .sync_index) if False else None
        # Use the selected candidate's person_ts to recover the selected human entry sync index for logging.
        sel_sync = None
        for e in self.history.get(cam, []):
            if int(e.event_ts_us or 0) == int(best.person_ts):
                sel_sync = int(e.sync_index)
                break
        d["person_frame_sync_index"] = sel_sync
        d["event_to_mvs_homography"] = "configured"
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.out_files[cam].write(line)
        self.all_fp.write(line)
        self.cand_count += 1
        print(f"[hit_judge][{cam}] impact candidate bullet={best.bullet_id} person={best.stable_id} score={best.total_score:.3f} offset={d.get('impact_offset_ms','?')}ms dt={best.dt_ms:.1f}ms method={best.match_method} strategy={strategy_used}", flush=True)
        return "emitted"
'''
s=s[:start]+new_block+s[end:]
# fix run stats line
s=s.replace('''                    print(f"[hit_judge][perf] recv={self.recv_count} cand={self.cand_count} no_human={self.no_human_count} no_cam={self.no_cam_count} {hist_info}", flush=True)
                    self.recv_count = self.cand_count = self.no_human_count = self.no_cam_count = 0
''','''                    print(f"[hit_judge][perf] recv={self.recv_count} cand={self.cand_count} wait={self.wait_count} drop={self.drop_count} nonimpact={self.nonimpact_count} no_human={self.no_human_count} no_cam={self.no_cam_count} pending={len(self.pending_events)} {hist_info}", flush=True)
                    self.recv_count = self.cand_count = self.no_human_count = self.no_cam_count = self.nonimpact_count = self.wait_count = self.drop_count = 0
''')
# add process_pending_events in run after receive_udp
s=s.replace('''                self.poll_inputs()
                self.receive_udp()
                now = time.time()
''','''                self.poll_inputs()
                self.receive_udp()
                self.process_pending_events()
                now = time.time()
''')
# update parse args desc no necessary
p.write_text(s, encoding='utf-8')
# copy to rentijiance
Path('/mnt/data/impact_hit_work/rentijiance/hit_judge_server.py').write_text(s, encoding='utf-8')
