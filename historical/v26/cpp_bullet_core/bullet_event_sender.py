"""
bullet_event_sender.py  —  子弹事件异步 UDP 发送器

职责：
  1. BulletEventSender：维护一个小队列和一个 daemon 发送线程，
     主循环只调 submit()，不做任何网络 I/O，保证 3ms 主循环不被阻塞。

  2. BulletEventExtractor：从 LinearMotionFilter.get_last_debug_rows()
     的逐帧输出中识别 turn_event 和 terminate_event，构造 BulletEvent
     并提交给 BulletEventSender。

  3. approach_point 取法：
     从 debug_rows 历史缓存中，反向遍历当前 bullet 的轨迹点，
     跳过与 event_point 距离 < 2 * min_step_px 的点，
     取第一个满足条件的点作为 approach_point。
     找不到时 approach_valid=False，命中模块退化为点判定。

使用方式（在主脚本 main() 里）：

    from bullet_event_sender import BulletEventSender, BulletEventExtractor

    # 初始化（在 line_filter 创建后）
    bullet_sender    = BulletEventSender(server_ip, server_port, enabled=True)
    bullet_extractor = BulletEventExtractor(
        sender=bullet_sender,
        camera_sn=args.tsf1_camera_id,  # --tsf1-camera-id 直接填序列号，如 4110046947
        orig_w=width,
        orig_h=height,
        min_step_px=args.line_min_step_px,
    )
    bullet_sender.start()

    # 主循环里 line_filter.update() 之后
    accepted_clusters = line_filter.update(clusters_np, ts)
    debug_rows        = line_filter.get_last_debug_rows()
    bullet_extractor.process(debug_rows)

    # finally 里
    bullet_sender.close()
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

from hit_protocol import (
    BulletEvent,
    EVENT_TERMINATE,
    EVENT_TURN,
    REASON_UNKNOWN,
)


# ---------------------------------------------------------------------------
# BulletEventSender — 异步 UDP 发送器
# ---------------------------------------------------------------------------

class BulletEventSender:
    """
    子弹事件异步 UDP 发送器。

    主循环只调 submit()，发送线程在后台异步消费队列。
    terminate_event 会在初次发送后延迟 RETRANSMIT_DELAY_MS 毫秒重发一次。
    """

    RETRANSMIT_DELAY_MS = 15   # terminate_event 重发间隔（毫秒）
    QUEUE_MAX_SIZE      = 512  # 内部队列最大容量；保留更多 track_probe/terminate 证据

    def __init__(self, server_ip: str, server_port: int, enabled: bool = True):
        self.server_ip   = server_ip
        self.server_port = int(server_port)
        self.enabled     = enabled

        self._queue:  queue.Queue = queue.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._worker, daemon=True, name="BulletEventSender")
        self._sock:   Optional[socket.socket] = None
        self._sent_count   = 0
        self._drop_count   = 0
        self._last_err_ts  = 0.0
        self._last_drop_warn_ts = 0.0
        self._drop_by_type: Dict[str, int] = {}

    def start(self) -> None:
        if not self.enabled:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread.start()
        print(f"[BulletEventSender] started → {self.server_ip}:{self.server_port}")

    def submit(self, event: BulletEvent) -> None:
        """
        把事件放入队列，主循环调用此方法，不做网络 I/O。

        队列满时优先保留 terminate / track_probe 类证据：
          - 新来的 terminate 类事件会尝试挤掉队列里最早的非 terminate 事件；
          - 新来的普通 turn_event 不再挤掉旧 terminate，直接丢弃；
          - 丢包类型会计数并低频打印，便于后面排查 false negative。
        """
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass

        if event.event_type == EVENT_TERMINATE and self._drop_one_queued_non_terminate():
            try:
                self._queue.put_nowait(event)
                return
            except queue.Full:
                pass

        self._mark_drop(event)

    def _drop_one_queued_non_terminate(self) -> bool:
        """Drop the oldest queued non-terminate event, preserving high-value evidence."""
        try:
            with self._queue.mutex:
                q = self._queue.queue
                for idx, old in enumerate(list(q)):
                    if getattr(old, "event_type", None) != EVENT_TERMINATE:
                        try:
                            del q[idx]
                            self._queue.unfinished_tasks = max(0, self._queue.unfinished_tasks - 1)
                        except Exception:
                            return False
                        self._mark_drop(old)
                        return True
        except Exception:
            return False
        return False

    def _mark_drop(self, event: BulletEvent) -> None:
        self._drop_count += 1
        key = str(getattr(event, "event_type", "unknown") or "unknown")
        self._drop_by_type[key] = int(self._drop_by_type.get(key, 0)) + 1
        now = time.time()
        if now - self._last_drop_warn_ts > 1.0:
            print(f"[BulletEventSender][WARN] queue full/drop total={self._drop_count} by_type={self._drop_by_type}")
            self._last_drop_warn_ts = now

    def close(self) -> None:
        self._stop.set()
        if self.enabled:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        print(f"[BulletEventSender] closed. sent={self._sent_count} dropped={self._drop_count} by_type={self._drop_by_type}")

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _send_packet(self, event: BulletEvent) -> None:
        """发送单个事件包，失败时打印错误（限频 1 次/秒）。"""
        try:
            payload = event.to_json()
            self._sock.sendto(payload, (self.server_ip, self.server_port))
            self._sent_count += 1
        except Exception as e:
            now = time.time()
            if now - self._last_err_ts > 1.0:
                print(f"[BulletEventSender] send error: {e}")
                self._last_err_ts = now

    def _worker(self) -> None:
        """发送线程：从队列取事件，发出，terminate_event 延迟重发。"""
        retransmit_queue: List[Tuple[float, BulletEvent]] = []  # (send_at_wall, event)

        while not self._stop.is_set():
            # 先处理到期的重发
            now = time.time()
            still_pending = []
            for send_at, ev in retransmit_queue:
                if now >= send_at:
                    retransmit_ev = BulletEvent(**{
                        **ev.__dict__,
                        "is_retransmit": True,
                    })
                    self._send_packet(retransmit_ev)
                else:
                    still_pending.append((send_at, ev))
            retransmit_queue = still_pending

            # 取新事件
            try:
                event = self._queue.get(timeout=0.005)  # 5ms 等待
            except queue.Empty:
                continue

            self._send_packet(event)

            # terminate_event 排入重发队列
            if event.event_type == EVENT_TERMINATE and not event.is_retransmit:
                send_at = time.time() + self.RETRANSMIT_DELAY_MS / 1000.0
                retransmit_queue.append((send_at, event))


# ---------------------------------------------------------------------------
# BulletEventExtractor — 从 debug_rows 中提取子弹事件
# ---------------------------------------------------------------------------

class BulletEventExtractor:
    """
    从 LinearMotionFilter.get_last_debug_rows() 的输出中提取事件。

    在 update() 外层主循环中调用 process(debug_rows)，不修改过滤器内部逻辑。

    状态追踪：
      _prev_bullet_active  : 上一帧各 stable_track_id 的 bullet_active 状态
      _bullet_seq_ids      : 每个 bullet_id 的 seq_id 计数器
      _bullet_sent_turns   : 已发送的 (bullet_id, segment_index) 集合，防止重发
      _point_history       : 最近若干帧各 bullet_id 的坐标历史，用于取 approach_point
    """

    POINT_HISTORY_MAXLEN = 64  # 保留更多点，用于连续直线轨迹质量判断
    BULLET_LIFETIME_US   = 300_000  # 【新增】子弹最大存活时间 (300ms)，超过此时间的轨迹视为无效

    # ---- 新增：低频 track_probe，解决“有 bullet_effect 轨迹但没有 turn/terminate UDP”的问题 ----
    # 每条已经进入显示层的 bullet 最多发一次 probe。probe 仍走 terminate 事件类型，
    # terminate_reason="track_probe"，所以不需要改 UDP 协议，也不会恢复高频 path_sample。
    TRACK_PROBE_PHASES = {"probation_passed", "maintain", "maintain_turn"}
    TRACK_PROBE_REASON = "track_probe"
    TRACK_PROBE_MIN_POINTS = 3
    TRACK_PROBE_MIN_PATH_LEN_PX = 45.0
    TRACK_PROBE_MIN_DURATION_MS = 12.0
    TRACK_PROBE_MIN_STRAIGHTNESS = 0.75
    TRACK_PROBE_MAX_LINE_ERROR_PX = 12.0
    TRACK_PROBE_MIN_SPEED_PX_PER_MS = 0.5
    TRACK_PROBE_MIN_APPROACH_LEN_PX = 45.0
    # track_probe 不再“每条子弹只发一次”。真实击中点可能出现在长轨迹后段；
    # 如果第一包 probe 发得太早，只会检查轨迹起点附近，后续真正穿过人体 mask 时就没有事件了。
    # 因此改为低频、限量、按位移触发的连续 probe：约 40ms 一次，每条最多 12 次。
    TRACK_PROBE_MIN_INTERVAL_US = 40_000
    TRACK_PROBE_MIN_MOVE_PX = 35.0
    TRACK_PROBE_MAX_PER_BULLET = 12
    TRACK_PROBE_RECENT_POINTS = 12
    TRACK_PROBE_RECENT_MAX_AGE_US = 140_000

    # ---- 修复：最低存活过滤，防止极短寿命的误判子弹发出 terminate 事件 ----
    TERMINATE_MIN_LIFETIME_FRAMES = 3   # bullet 从激活到终止必须存活至少 N 帧才发 terminate
    TERMINATE_MIN_LIFETIME_US     = 12_000  # bullet 存活时间 < 12ms 不发 terminate

    def __init__(
        self,
        sender:       BulletEventSender,
        camera_sn:    str,            # 相机唯一硬件序列号（如 "4110046947"）
        camera_alias: str  = "",      # 可选显示名
        orig_w:       int  = 0,
        orig_h:       int  = 0,
        min_step_px:  float = 3.0,
    ):
        self.sender       = sender
        self.camera_sn    = camera_sn
        self.camera_alias = camera_alias
        self.orig_w      = int(orig_w)
        self.orig_h      = int(orig_h)
        self.min_step_px = float(min_step_px)

        # 上一帧状态
        self._prev_bullet_active:  Dict[int, bool]  = {}   # stable_track_id → bool
        self._prev_bullet_id:      Dict[int, int]   = {}   # stable_track_id → bullet_id

        # ---- 修复：记录每个 bullet 的激活时间和帧数，用于 terminate 最低存活过滤 ----
        self._bullet_activate_ts:    Dict[int, int]  = {}   # bullet_id → 激活时的 timestamp_us
        self._bullet_activate_frames: Dict[int, int] = {}   # bullet_id → 激活后经历的帧数

        # 每个 bullet_id 的 seq_id（全程递增）
        self._bullet_seq_ids:      Dict[int, int]   = {}   # bullet_id → next_seq_id

        # 已发送的 turn 事件，防止重复发（同一 bullet_id + segment_index 只发一次）
        self._sent_turns:          set              = set()  # (bullet_id, segment_index)

        # 已发送的 track_probe 状态。
        # 旧版本每条 display bullet 只发一次，容易在轨迹刚开始时过早发送，
        # 后续真正穿过人体 mask 时已经不会再触发 hit_judge。
        # 这里改为“低频多次”：每条最多 TRACK_PROBE_MAX_PER_BULLET 次，
        # 且必须间隔足够时间、移动足够距离，避免恢复高频 path_sample。
        self._sent_track_probes:   set              = set()  # 兼容旧清理逻辑；不再作为唯一发送门控
        self._track_probe_state:   Dict[int, Dict[str, float]] = {}

        # 坐标历史，用于找 approach_point
        # {bullet_id: [(cx, cy, timestamp_us), ...]}  最新在末尾
        self._point_history:       Dict[int, List[Tuple[float, float, int]]] = {}

        # P11/P12: 只有已经正式进入显示层(display_bullet_id>=0)的 bullet，
        # 才允许对外发送 turn/terminate 事件。pending 候选只保留内部历史，
        # 防止人体快速运动/撞击残留触发误命中。
        self._event_eligible_bullets: set = set()

        # Per-slice debug context injected by the event pipeline before process().
        # Example: current-frame human-mask filter stats.  The context is copied
        # into sparse BulletEvent packets so hit_judge can write it into debug JSONL.
        self._event_context: Dict[str, object] = {}

    def set_event_context(self, context: Optional[Dict[str, object]]) -> None:
        self._event_context = dict(context or {})

    def _current_human_mask_stats(self) -> Dict[str, object]:
        stats = self._event_context.get("event_human_mask_filter", {})
        return dict(stats) if isinstance(stats, dict) else {}

    # -----------------------------------------------------------------------
    # 对外接口
    # -----------------------------------------------------------------------

    def process(self, debug_rows: List[dict]) -> None:
        """
        每次 line_filter.update() 后调用，传入 get_last_debug_rows() 的结果。
        内部识别 turn 和 terminate 事件并提交给 sender。
        """
        if not debug_rows:
            return

        for row in debug_rows:
            bullet_id   = int(row.get("bullet_id", -1))
            display_id  = int(row.get("display_bullet_id", -1))
            if bullet_id > 0 and display_id >= 0:
                self._event_eligible_bullets.add(int(bullet_id))
            stable_id   = int(row.get("stable_track_id", -1))
            active      = bool(int(row.get("bullet_active", 0)))
            phase       = str(row.get("phase", ""))
            seg_idx     = int(row.get("segment_index", 0))
            timestamp_us = int(row.get("timestamp", 0))
            # P8: 命中事件坐标优先使用模型轨迹坐标；cx/cy 只是检测框中心观测。
            cx          = float(row.get("model_x", row.get("cx", 0.0)))
            cy          = float(row.get("model_y", row.get("cy", 0.0)))
            cum_deg     = float(row.get("cumulative_turn_deg", 0.0))
            reason      = str(row.get("reject_reason", REASON_UNKNOWN))
            speed       = self._estimate_speed(bullet_id, cx, cy, timestamp_us)

            # 更新坐标历史（每帧都记，不管 bullet_active）
            if bullet_id > 0:
                self._update_point_history(bullet_id, cx, cy, timestamp_us)

            prev_active = self._prev_bullet_active.get(stable_id, False)
            prev_bid    = self._prev_bullet_id.get(stable_id, -1)

            # ---- 修复：跟踪 bullet 激活时间和帧数 ----
            if active and bullet_id > 0:
                if bullet_id not in self._bullet_activate_ts:
                    # 首次激活
                    self._bullet_activate_ts[bullet_id] = timestamp_us
                    self._bullet_activate_frames[bullet_id] = 0
                self._bullet_activate_frames[bullet_id] = \
                    self._bullet_activate_frames.get(bullet_id, 0) + 1

            # ---- 新增：识别 track_probe ----
            # 真实测试中，bullet_effect_D.csv 可能已经出现完整轨迹，
            # 但该轨迹没有 maintain_turn，也没有稳定触发 active->inactive terminate。
            #
            # v1/v2 中每条 bullet 只发一次 probe，第一包通常发生在轨迹刚成立的起点附近；
            # 如果真正击中点在轨迹后半段，hit_judge 只检查了起点附近，自然会漏判。
            # 这里改为低频多次 probe：同一 bullet 最多发 TRACK_PROBE_MAX_PER_BULLET 次，
            # 且两次 probe 之间必须间隔足够时间、移动足够距离。
            if (active and bullet_id > 0
                    and bullet_id in self._event_eligible_bullets
                    and phase in self.TRACK_PROBE_PHASES
                    and self._should_emit_track_probe(bullet_id, timestamp_us)):
                probe_speed = self._estimate_speed_from_history(bullet_id)
                ok, _reason, _quality, _approach_len = self._track_probe_quality_ok(bullet_id, probe_speed)
                if ok:
                    self._mark_track_probe_sent(bullet_id, timestamp_us, cx, cy)
                    self._emit_track_probe(
                        bullet_id    = bullet_id,
                        timestamp_us = timestamp_us,
                        cx=cx, cy=cy,
                        seg_idx      = seg_idx,
                        speed        = probe_speed,
                    )

            # ---- P19: C++ shot_bounce_link 作为一次 turn-like 事件发送 ----
            # C++ line filter 已经确认：新 bullet_id 片段是旧物理子弹的反弹后段，
            # 因此这里给 hit_judge 一个稀疏 turn 事件。注意：这不是 HIT 证明，
            # hit_judge 仍必须检查人体 mask 接触证据。
            birth_kind = str(row.get("birth_assign_kind", ""))
            shot_link_type = str(row.get("shot_link_type", ""))
            if (active and bullet_id > 0
                    and bullet_id in self._event_eligible_bullets
                    and int(row.get("just_assigned_bullet", 0))
                    and ("shot_bounce_link" in birth_kind or shot_link_type == "bounce")
                    and (bullet_id, seg_idx) not in self._sent_turns):
                self._sent_turns.add((bullet_id, seg_idx))
                self._emit_shot_link_turn(
                    bullet_id    = bullet_id,
                    timestamp_us = timestamp_us,
                    cx=cx, cy=cy,
                    seg_idx      = seg_idx,
                    turn_angle_deg = float(row.get("shot_link_angle_deg", 0.0) or 0.0),
                    speed        = speed,
                    parent_x     = float(row.get("shot_link_parent_x", cx) or cx),
                    parent_y     = float(row.get("shot_link_parent_y", cy) or cy),
                    shot_link_type = shot_link_type or "bounce",
                    shot_link_reason = str(row.get("shot_link_reason", "p19_cpp_shot_bounce_link")),
                    parent_bullet_id = int(row.get("shot_link_parent_bullet_id", -1) or -1),
                    parent_segment_index = int(row.get("shot_link_parent_segment_index", -1) or -1),
                    gap_ms = float(row.get("shot_link_gap_ms", 0.0) or 0.0),
                    gap_dist_px = float(row.get("shot_link_gap_dist_px", 0.0) or 0.0),
                )

            # ---- 识别 turn_event ----
            if (active and bullet_id > 0
                    and bullet_id in self._event_eligible_bullets
                    and phase == "maintain_turn"
                    and (bullet_id, seg_idx) not in self._sent_turns):
                # 【新增】检查子弹是否已超时，防止陈旧的轨迹数据干扰
                if timestamp_us - self._point_history.get(bullet_id, [(-1, -1, 0)])[-1][2] > self.BULLET_LIFETIME_US:
                    continue
                self._sent_turns.add((bullet_id, seg_idx))
                self._emit_turn(
                    bullet_id    = bullet_id,
                    timestamp_us = timestamp_us,
                    cx=cx, cy=cy,
                    seg_idx      = seg_idx,
                    turn_angle_deg = self._calc_turn_angle(cum_deg, seg_idx, bullet_id),
                    speed        = speed,
                )

            # ---- 识别 terminate_event ----
            # 条件：上一帧 bullet_active=True，本帧 bullet_active=False
            # 用 prev_bullet_id 找终止的是哪个 bullet
            if prev_active and not active and prev_bid > 0:
                if prev_bid not in self._event_eligible_bullets:
                    # pending/未显示候选终止：不对外发事件，只清理内部状态。
                    self._bullet_activate_ts.pop(prev_bid, None)
                    self._bullet_activate_frames.pop(prev_bid, None)
                    self._point_history.pop(prev_bid, None)
                    self._sent_turns = {k for k in self._sent_turns if k[0] != prev_bid}
                    self._sent_track_probes.discard(prev_bid)
                    self._track_probe_state.pop(prev_bid, None)
                    self._event_eligible_bullets.discard(prev_bid)
                    self._prev_bullet_active[stable_id] = active
                    self._prev_bullet_id[stable_id] = bullet_id if bullet_id > 0 else prev_bid
                    continue
                # ---- 修复：最低存活过滤 ----
                # 检查这个 bullet 从激活到终止经历了多少帧和多长时间
                # 太短寿命的 bullet 大概率是噪声误判，不应发出 terminate 事件
                activate_ts     = self._bullet_activate_ts.get(prev_bid, timestamp_us)
                activate_frames = self._bullet_activate_frames.get(prev_bid, 0)
                lifetime_us     = timestamp_us - activate_ts

                if (activate_frames < self.TERMINATE_MIN_LIFETIME_FRAMES
                        or lifetime_us < self.TERMINATE_MIN_LIFETIME_US):
                    # 存活太短，抑制 terminate 事件

                    # 清理该 bullet 的记录
                    self._bullet_activate_ts.pop(prev_bid, None)
                    self._bullet_activate_frames.pop(prev_bid, None)
                    self._point_history.pop(prev_bid, None)
                    self._sent_turns = {k for k in self._sent_turns if k[0] != prev_bid}
                    self._sent_track_probes.discard(prev_bid)
                    self._track_probe_state.pop(prev_bid, None)
                    self._event_eligible_bullets.discard(prev_bid)
                else:
                    # 从 prev_bid 的坐标历史取真实终止点坐标和时间戳
                    # 不用当前帧的 cx/cy，因为当前帧 bullet_active=0，已属于其他目标
                    _hist = self._point_history.get(prev_bid, [])
                    if _hist:
                        term_cx, term_cy, term_ts = _hist[-1]
                    else:
                        term_cx, term_cy, term_ts = cx, cy, timestamp_us
                    term_speed  = self._estimate_speed_from_history(prev_bid)
                    term_reason = self._map_terminate_reason(reason)
                    self._emit_terminate(
                        bullet_id    = prev_bid,
                        timestamp_us = term_ts,
                        cx=term_cx, cy=term_cy,
                        seg_idx      = int(row.get("segment_index", 0)),
                        reason       = term_reason,
                        speed        = term_speed,
                    )
                    # 清理激活记录
                    self._bullet_activate_ts.pop(prev_bid, None)
                    self._bullet_activate_frames.pop(prev_bid, None)

            # 更新前帧状态
            self._prev_bullet_active[stable_id] = active
            self._prev_bullet_id[stable_id]     = bullet_id if bullet_id > 0 else prev_bid

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    def _next_seq_id(self, bullet_id: int) -> int:
        seq = self._bullet_seq_ids.get(bullet_id, 0)
        self._bullet_seq_ids[bullet_id] = seq + 1
        return seq

    def _update_point_history(self, bullet_id: int, cx: float, cy: float, ts: int) -> None:
        if bullet_id not in self._point_history:
            self._point_history[bullet_id] = []
        hist = self._point_history[bullet_id]
        # 只在坐标发生变化时追加（避免静止点污染）
        if hist and abs(hist[-1][0] - cx) < 0.1 and abs(hist[-1][1] - cy) < 0.1:
            return
        hist.append((cx, cy, ts))
        # 保留最近 N 帧
        if len(hist) > self.POINT_HISTORY_MAXLEN:
            self._point_history[bullet_id] = hist[-self.POINT_HISTORY_MAXLEN:]

    def _recent_quality_history(
        self, hist: List[Tuple[float, float, int]]
    ) -> List[Tuple[float, float, int]]:
        """
        返回用于 track_quality / track_probe 的近期轨迹窗口。

        旧逻辑用整条 bullet 历史计算直线性和 line_error。实测中如果两发很近、
        或同一 bullet_id 被误合并成一条长轨迹，整条历史会变弯，
        后半段即使是清晰直线也会因为全局 line_error 过大被 hit_judge 拦掉。
        这里用最近若干点 / 最近 140ms 的局部直线段评价质量，更符合“当前击中点”。
        """
        if not hist:
            return []
        max_points = int(self.TRACK_PROBE_RECENT_POINTS)
        max_age_us = int(self.TRACK_PROBE_RECENT_MAX_AGE_US)
        recent = list(hist[-max_points:])
        if recent:
            latest_ts = int(recent[-1][2])
            recent = [p for p in recent if latest_ts - int(p[2]) <= max_age_us]
        return recent if recent else list(hist[-max_points:])


    def _should_emit_track_probe(self, bullet_id: int, timestamp_us: int) -> bool:
        """低频、限量地发送 track_probe，避免恢复高频 path_sample。"""
        state = self._track_probe_state.get(int(bullet_id))
        if not state:
            return True

        count = int(state.get("count", 0) or 0)
        if count >= int(self.TRACK_PROBE_MAX_PER_BULLET):
            return False

        last_ts = int(state.get("last_ts", 0) or 0)
        if int(timestamp_us) - last_ts < int(self.TRACK_PROBE_MIN_INTERVAL_US):
            return False

        hist = self._point_history.get(int(bullet_id), [])
        if not hist:
            return False
        cx, cy, _ts = hist[-1]
        last_x = float(state.get("last_x", cx) or cx)
        last_y = float(state.get("last_y", cy) or cy)
        move = ((float(cx) - last_x) ** 2 + (float(cy) - last_y) ** 2) ** 0.5
        if move < float(self.TRACK_PROBE_MIN_MOVE_PX):
            return False
        return True


    def _mark_track_probe_sent(self, bullet_id: int, timestamp_us: int, cx: float, cy: float) -> None:
        state = self._track_probe_state.get(int(bullet_id), {})
        count = int(state.get("count", 0) or 0) + 1
        self._track_probe_state[int(bullet_id)] = {
            "count": int(count),
            "last_ts": int(timestamp_us),
            "last_x": float(cx),
            "last_y": float(cy),
        }
        self._sent_track_probes.add(int(bullet_id))


    def _track_quality(self, bullet_id: int) -> Dict[str, float]:
        """
        计算一个 bullet_id 的连续轨迹质量。

        这些字段只在 turn/terminate 事件里随 UDP 发出，不额外增加发送频率。
        用于 hit_judge 过滤人体边缘/衣服/反光产生的伪 bullet。
        """
        hist_all = self._point_history.get(bullet_id, [])
        hist = self._recent_quality_history(hist_all)
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


    def _track_probe_quality_ok(
        self, bullet_id: int, speed_px_per_ms: float
    ) -> Tuple[bool, str, Dict[str, float], float]:
        """
        判断当前 bullet 是否足够像一条真实飞行轨迹，可以发一次 track_probe。

        这个门控只控制是否发 probe，不影响原有 turn/terminate 的发送。
        阈值故意与 hit_judge 的默认安全阈值保持一致：点数、路径长度、持续时间、
        直线性、线误差、速度和 approach 长度都要基本合格。
        """
        quality = self._track_quality(bullet_id)
        hist = self._point_history.get(bullet_id, [])
        if hist:
            event_x, event_y, _ts = hist[-1]
            _ax, _ay, approach_len, approach_valid = self._find_approach_point(
                bullet_id, float(event_x), float(event_y)
            )
        else:
            approach_len, approach_valid = 0.0, False

        point_count = int(quality.get("track_point_count", 0) if quality.get("track_point_count", 0) is not None else 0)
        path_len = float(quality.get("track_path_len_px", 0.0) if quality.get("track_path_len_px", 0.0) is not None else 0.0)
        duration_ms = float(quality.get("track_duration_ms", 0.0) if quality.get("track_duration_ms", 0.0) is not None else 0.0)
        straightness = float(quality.get("track_straightness", 0.0) if quality.get("track_straightness", 0.0) is not None else 0.0)
        line_error = float(quality.get("track_line_error_px", 999.0) if quality.get("track_line_error_px", 999.0) is not None else 999.0)
        speed = float(speed_px_per_ms if speed_px_per_ms is not None else 0.0)

        if point_count < self.TRACK_PROBE_MIN_POINTS:
            return False, "track_probe_point_count_too_small", quality, approach_len
        if path_len < self.TRACK_PROBE_MIN_PATH_LEN_PX:
            return False, "track_probe_path_len_too_short", quality, approach_len
        if duration_ms < self.TRACK_PROBE_MIN_DURATION_MS:
            return False, "track_probe_duration_too_short", quality, approach_len
        if straightness < self.TRACK_PROBE_MIN_STRAIGHTNESS:
            return False, "track_probe_not_straight", quality, approach_len
        if line_error > self.TRACK_PROBE_MAX_LINE_ERROR_PX:
            return False, "track_probe_line_error_too_large", quality, approach_len
        if speed < self.TRACK_PROBE_MIN_SPEED_PX_PER_MS:
            return False, "track_probe_speed_too_low", quality, approach_len
        if (not approach_valid) or approach_len < self.TRACK_PROBE_MIN_APPROACH_LEN_PX:
            return False, "track_probe_approach_len_too_short", quality, approach_len
        return True, "", quality, approach_len


    def _find_approach_point(
        self, bullet_id: int, event_x: float, event_y: float
    ) -> Tuple[float, float, float, bool]:
        """
        从轨迹历史中找更早、更远的 approach_point。

        目的：
          用 approach_point -> event_point 这条较长轨迹段判断是否穿过人体 mask。
          不额外发送 path_sample，不增加 UDP 频率。
        """
        hist_all = self._point_history.get(bullet_id, [])
        hist = self._recent_quality_history(hist_all)
        if not hist:
            return event_x, event_y, 0.0, False

        # 至少拉出一段有意义的线段。太短的线段容易把人体局部噪声判成 HIT。
        # 使用近期窗口而不是整条历史，避免一条被合并的长弯轨迹用最早点跨过人体造成误判。
        min_segment_len = max(float(self.TRACK_PROBE_MIN_APPROACH_LEN_PX), 8.0 * float(self.min_step_px))

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

    def _estimate_speed(self, bullet_id: int, cx: float, cy: float, ts: int) -> float:
        """
        用最近几帧坐标历史估算平均速度（px/ms）。
        取历史最老点到当前点的位移 / 时间差。
        """
        hist = self._point_history.get(bullet_id, [])
        if len(hist) < 2:
            return 0.0
        # 取最多 4 帧前的点
        ref_idx = max(0, len(hist) - 4)
        ref_cx, ref_cy, ref_ts = hist[ref_idx]
        dt_us = ts - ref_ts
        if dt_us <= 0:
            return 0.0
        dist = ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
        return float(dist / (dt_us / 1000.0))  # px/ms

    def _estimate_speed_from_history(self, bullet_id: int) -> float:
        # 专门给 terminate_event 用：完全从 prev_bid 坐标历史估算速度
        hist = self._point_history.get(bullet_id, [])
        if len(hist) < 2:
            return 0.0
        ref_idx = max(0, len(hist) - 4)
        ref_cx, ref_cy, ref_ts = hist[ref_idx]
        last_cx, last_cy, last_ts = hist[-1]
        dt_us = last_ts - ref_ts
        if dt_us <= 0:
            return 0.0
        dist = ((last_cx - ref_cx) ** 2 + (last_cy - ref_cy) ** 2) ** 0.5
        return float(dist / (dt_us / 1000.0))

    def _calc_turn_angle(self, cum_deg: float, seg_idx: int, bullet_id: int) -> float:
        """
        从 cumulative_turn_deg 估算本次 turn 的角度。
        用当前累计角度减去上一个 segment 的累计角度。
        """
        key = f"{bullet_id}__prev_cum_deg"
        prev_cum = getattr(self, "_turn_cum_cache", {}).get(key, 0.0)
        if not hasattr(self, "_turn_cum_cache"):
            self._turn_cum_cache: Dict[str, float] = {}
        angle = max(0.0, cum_deg - prev_cum)
        self._turn_cum_cache[key] = cum_deg
        return float(angle)

    @staticmethod
    def _map_terminate_reason(reject_reason: str) -> str:
        """把 reject_reason 映射到标准的 terminate_reason 常量。"""
        mapping = {
            "maintain_offset":       "offset_exceed",
            "maintain_recent_offset": "recent_offset_exceed",
            "maintain_bbox":         "bbox_unstable",
            "maintain_bbox_ema":     "bbox_ema_drift",
            "maintain_overturn":     "overturn",
            "maintain_static":       "static",
            "missing":               "missing",
            "probation":             "probation",
            "probation_offset":      "probation_offset",
        }
        return mapping.get(reject_reason, REASON_UNKNOWN)

    def _emit_shot_link_turn(
        self,
        bullet_id: int,
        timestamp_us: int,
        cx: float, cy: float,
        seg_idx: int,
        turn_angle_deg: float,
        speed: float,
        parent_x: float, parent_y: float,
        shot_link_type: str,
        shot_link_reason: str,
        parent_bullet_id: int,
        parent_segment_index: int,
        gap_ms: float,
        gap_dist_px: float,
    ) -> None:
        approach_x = float(parent_x)
        approach_y = float(parent_y)
        dx = float(cx) - approach_x
        dy = float(cy) - approach_y
        approach_len = float((dx * dx + dy * dy) ** 0.5)
        track_quality = self._track_quality(bullet_id)
        event = BulletEvent(
            camera_sn            = self.camera_sn,
            camera_alias         = self.camera_alias,
            bullet_id            = bullet_id,
            seq_id               = self._next_seq_id(bullet_id),
            is_retransmit        = False,
            event_type           = EVENT_TURN,
            timestamp_us         = timestamp_us,
            event_x              = cx,
            event_y              = cy,
            approach_x           = approach_x,
            approach_y           = approach_y,
            approach_len_px      = approach_len,
            approach_valid       = approach_len >= max(8.0, float(self.min_step_px) * 2.0),
            orig_w               = self.orig_w,
            orig_h               = self.orig_h,
            avg_speed_px_per_ms  = speed,
            segment_index        = seg_idx,
            turn_angle_deg       = max(float(turn_angle_deg or 0.0), 1.0),
            terminate_reason     = REASON_UNKNOWN,
            confidence           = 1.0,
            shot_link_type       = str(shot_link_type or "bounce"),
            shot_link_reason     = str(shot_link_reason or "p19_cpp_shot_bounce_link"),
            shot_link_parent_bullet_id = int(parent_bullet_id),
            shot_link_parent_segment_index = int(parent_segment_index),
            shot_link_gap_ms     = float(gap_ms),
            shot_link_gap_dist_px = float(gap_dist_px),
            shot_link_angle_deg  = float(turn_angle_deg or 0.0),
            event_human_mask_stats = self._current_human_mask_stats(),
            **track_quality,
        )
        self.sender.submit(event)

    def _emit_turn(
        self,
        bullet_id:    int,
        timestamp_us: int,
        cx: float, cy: float,
        seg_idx:      int,
        turn_angle_deg: float,
        speed:        float,
    ) -> None:
        approach_x, approach_y, approach_len, approach_valid = \
            self._find_approach_point(bullet_id, cx, cy)
        track_quality = self._track_quality(bullet_id)

        event = BulletEvent(
            camera_sn            = self.camera_sn,
            camera_alias         = self.camera_alias,
            bullet_id            = bullet_id,
            seq_id               = self._next_seq_id(bullet_id),
            is_retransmit        = False,
            event_type           = EVENT_TURN,
            timestamp_us         = timestamp_us,
            event_x              = cx,
            event_y              = cy,
            approach_x           = approach_x,
            approach_y           = approach_y,
            approach_len_px      = approach_len,
            approach_valid       = approach_valid,
            orig_w               = self.orig_w,
            orig_h               = self.orig_h,
            avg_speed_px_per_ms  = speed,
            segment_index        = seg_idx,
            turn_angle_deg       = turn_angle_deg,
            terminate_reason     = REASON_UNKNOWN,
            confidence           = 1.0,
            event_human_mask_stats = self._current_human_mask_stats(),
            **track_quality,
        )
        self.sender.submit(event)

    def _emit_track_probe(
        self,
        bullet_id:    int,
        timestamp_us: int,
        cx: float, cy: float,
        seg_idx:      int,
        speed:        float,
    ) -> None:
        """
        发送一次低频轨迹探针。

        设计原则：
          - 使用 EVENT_TERMINATE + terminate_reason="track_probe"，兼容现有协议；
          - 不清理 point_history / sent_turns / eligible 状态，避免破坏后续真实 turn/terminate；
          - 每个 bullet_id 由 process() 保证最多发送一次。
        """
        approach_x, approach_y, approach_len, approach_valid = \
            self._find_approach_point(bullet_id, cx, cy)
        track_quality = self._track_quality(bullet_id)

        event = BulletEvent(
            camera_sn            = self.camera_sn,
            camera_alias         = self.camera_alias,
            bullet_id            = bullet_id,
            seq_id               = self._next_seq_id(bullet_id),
            is_retransmit        = False,
            event_type           = EVENT_TERMINATE,
            timestamp_us         = timestamp_us,
            event_x              = cx,
            event_y              = cy,
            approach_x           = approach_x,
            approach_y           = approach_y,
            approach_len_px      = approach_len,
            approach_valid       = approach_valid,
            orig_w               = self.orig_w,
            orig_h               = self.orig_h,
            avg_speed_px_per_ms  = speed,
            segment_index        = seg_idx,
            turn_angle_deg       = 0.0,
            terminate_reason     = self.TRACK_PROBE_REASON,
            confidence           = 1.0,
            event_human_mask_stats = self._current_human_mask_stats(),
            **track_quality,
        )
        self.sender.submit(event)

    def _emit_terminate(
        self,
        bullet_id:    int,
        timestamp_us: int,
        cx: float, cy: float,
        seg_idx:      int,
        reason:       str,
        speed:        float,
    ) -> None:
        approach_x, approach_y, approach_len, approach_valid = \
            self._find_approach_point(bullet_id, cx, cy)
        track_quality = self._track_quality(bullet_id)

        event = BulletEvent(
            camera_sn            = self.camera_sn,
            camera_alias         = self.camera_alias,
            bullet_id            = bullet_id,
            seq_id               = self._next_seq_id(bullet_id),
            is_retransmit        = False,
            event_type           = EVENT_TERMINATE,
            timestamp_us         = timestamp_us,
            event_x              = cx,
            event_y              = cy,
            approach_x           = approach_x,
            approach_y           = approach_y,
            approach_len_px      = approach_len,
            approach_valid       = approach_valid,
            orig_w               = self.orig_w,
            orig_h               = self.orig_h,
            avg_speed_px_per_ms  = speed,
            segment_index        = seg_idx,
            turn_angle_deg       = 0.0,
            terminate_reason     = reason,
            confidence           = 1.0,
            event_human_mask_stats = self._current_human_mask_stats(),
            **track_quality,
        )
        self.sender.submit(event)

        # 终止后清理该 bullet 的坐标历史和 turn 记录（节省内存）
        self._point_history.pop(bullet_id, None)
        self._sent_turns   = {k for k in self._sent_turns if k[0] != bullet_id}
        self._sent_track_probes.discard(bullet_id)
        self._track_probe_state.pop(bullet_id, None)
        self._event_eligible_bullets.discard(bullet_id)
