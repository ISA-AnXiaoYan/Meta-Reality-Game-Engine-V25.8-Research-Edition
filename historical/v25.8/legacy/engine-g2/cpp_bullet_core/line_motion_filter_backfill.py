"""
line_motion_filter_backfill.py  —  直线运动过滤器（P14 raw_id handoff fix）（P10-7 merge：P10-5/P10-6 + early display promotion 修订版）

【P4 新增改动】（相对 P3 版，基于 75K 行 CSV 数据分析）

【P4-1】收紧 trigger 方向一致性内部系数
    _trigger_min_dir_ratio  : max(0.55, thresh*0.75) → max(0.72, thresh*0.90)
    _trigger_min_same_sign  : max(0.52, thresh*0.78) → max(0.72, thresh*0.90)
    当 thresh=0.80 时：内部门槛从 0.60 提高到 0.72。
    基于 CSV 数据：b1(dir_ok=0.67)、b11(dir_ok=0.4)、b15(dir_ok=0.4) 均为人体误判，
    真实子弹 dir_ok 始终 = 1.0。收紧后真实子弹不受影响，低一致性误判全部拦截。

【P4-2】提高 bullet_min_output_streak 最小持续帧数
    TrackConfig.bullet_min_output_streak: 2 → 5
    基于 CSV 数据：b5(1帧)、b21(1帧)、b3(4帧)、b12(5帧) 均为瞬时误判，
    真实子弹至少存活 34+ 帧。提高到 5 帧可消灭所有 1-3 帧的瞬时误判。

【P3 保留改动】（参数值调整见下方文档）

【P3 新增改动】（相对 P2 版）

【P3-1】大幅缩短续接时间/距离窗口
    _same_track_reactivate_window_us : 1_200_000 → 100_000（100ms）
    _ghost_reactivate_window_us      : 3_000_000 → 200_000（200ms）
    ghost_capture 内 dt 上限         : 300_000   → 100_000（100ms）
    ghost_bounce_max_dist_px 默认值  : 120.0 → 35.0
    _late_trigger_exempt_time_us     : 400_000 → 150_000
    _late_trigger_exempt_bounce_px   : 固定 30.0（不再随 bounce_max 缩放）
    理由：子弹弹道生命周期不超过 200ms，秒级窗口会把人体持续抖动串成同一 id。

【P3-2】反弹续接加速度合理性约束
    bounce 分支新增：dist_to_term / dt >= ghost_bounce_min_speed_px_per_us（默认 0.05）。
    人体静止后重现隐含速度接近 0；真实弹道续接速度远高于此。
    新增 TrackConfig.ghost_bounce_min_speed_px_per_us（默认 0.05）。

【P3-3】trigger 阶段加 full_stats max_offset 硬约束
    新增：full_stats['max_offset'] <= max_line_offset_px * trigger_max_full_offset_ratio。
    P3 初始 ratio=1.5（4.5px），P4 收紧到 1.0（3.0px）。
    基于 CSV 数据：b5(4.16px)、b15(4.35px)、b25(4.47px)、b26(3.92px) 均为人体误判，
    真实子弹 max_offset 始终 < 2.0px。收紧后真实子弹不受影响，误判全部拦截。
    新增 TrackConfig.trigger_max_full_offset_ratio（默认 1.0）。

【P3-4】P2-F 疑似续接豁免收紧
    豁免前加前置条件：full_stats['max_offset'] <= max_line_offset_px * 1.2。
    几何质量不干净的目标不给豁免。豁免路径不修改 direction_ok 门槛。

【P3-5】_slice_stats 新增 sign_flip_ratio；trigger 加翻转比率限制
    sign_flip_ratio：相邻步 dot < 0（方向反转）的比例。
    trigger 新增条件：sign_flip_ratio <= trigger_max_sign_flip_ratio。
    P3 初始 0.35，P4 收紧到 0.20（即 same_sign_ratio >= 0.80）。
    基于 CSV 数据：b4(flip=0.75)、b25(flip=0.71)、b26(flip=0.75) 均为人体抖动，
    真实子弹 flip 始终 = 1.0。收紧后真实子弹不受影响。
    失败时 reject_reason 追加 'flip'。
    新增 TrackConfig.trigger_max_sign_flip_ratio（默认 0.20）。

【P2 保留的改动】

【P2-A】maintain 阶段：recent_offset 连续超限快速 kill
    新增 TrackConfig.maintain_recent_offset_px（默认 4.0）和
    maintain_recent_offset_exceed_frames（默认 3）。
    在 maintain/probation 入口，若 recent_max_offset 连续超过该阈值达到
    指定帧数，则立即 deactivate，不再等 full_stats 的 SVD 慢慢拉高。

【P2-B】maintain 阶段：bbox EMA 慢漂移检测
    新增 TrackConfig.maintain_bbox_ema_alpha（默认 0.25，EMA 平滑系数）和
    maintain_bbox_ema_drift_ratio（默认 0.45，相对 EMA 均值的最大允许偏差率）。
    在 TrackState 中新增 bbox_ema_w / bbox_ema_h，随每帧更新。
    若当前帧 bbox 相对 EMA 均值的偏差率超过阈值，则立即 kill。
    与 8 帧 std 并行，两者任一触发即 kill。

【P2-C】maintain_turn 次数和累计角度限制
    新增 TrackConfig.maintain_max_turn_count（默认 2）和
    maintain_max_cumulative_turn_deg（默认 90.0）。
    TrackState 新增 turn_count / cumulative_turn_deg。
    每次 _update_segment_state 发生 turn 时累加。
    在 _evaluate_probation_and_maintain 最顶部：
    若 turn_count > max_turn_count 或 cumulative_turn_deg > max_cumulative_turn_deg，
    立即 kill（phase=terminated_overturn）。

【P2-D】turn 后 state.points 截断到当前段
    _update_segment_state 发生 turn 时，把 state.points 截断为只保留
    turn 点之后的新段点（最少保留 keep_tail_points 个），让 full_stats
    只反映当前段的直线质量，不再被历史段污染。
    display_points 不受影响，仍然保留完整轨迹用于绘制。

【P2-E】ghost 续接拆成"直飞续接"和"反弹续接"两条路径
    _try_ghost_capture_new_track 重构：
      - 直飞分支（forward）：与原来一致，沿旧方向前向外推，按 dist_to_pred 判断。
      - 反弹分支（bounce）：不做前向外推，直接用新观测点到旧终止点的原始距离判断，
        允许方向相反或垂直，距离门限单独控制（bounce_max_dist_px）。
    新增 TrackConfig.ghost_bounce_max_dist_px（默认 120.0）。
    反弹分支匹配成功时，score 附加小额惩罚以优先直飞分支，
    reuse_source 标记为 'ghost_capture_bounce' 方便调试。

【P2-F】疑似续接目标豁免 late_trigger
    _evaluate_trigger 中的 late_trigger_veto 增加例外条件：
    若该 track 的首观测点落在 recent_terminated 的时空窗口内（包含直飞和反弹
    两套距离标准），则不施加 late_trigger_veto，同时把 trigger 的 motion 要求
    降为 max(1, original-1)，给续接目标更低的运动门槛。
    为避免人体误触发后的 recent_terminated 被滥用，豁免时同时检查尺寸约束。

【P2-G】段内 state.points 静止点过滤（r116 类型）
    在 update() 主循环 append point 时，若新点与上一点距离 < min_step_px * 0.5，
    则跳过追加（不计入 state.points），避免静止起步段污染 valid_step_ratio。
    pre_confirm_points 和 display_points 仍正常追加，不受影响。

【P2-H】TrackState 新增诊断字段 / debug CSV 补全
    TrackState 新增：
      recent_offset_exceed_count（连续超 recent_offset 的帧数计数）
      bbox_ema_w / bbox_ema_h（EMA 均值）
      turn_count / cumulative_turn_deg（转向统计）
    debug_rows 新增字段：
      track_age_ms, path_over_net, recent_path_over_net,
      reuse_source, probation_passed, probation_fail_count,
      segment_index, compact_ok, turn_count, cumulative_turn_deg,
      recent_offset_exceed_count
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 参数 dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackConfig:
    """追踪几何与运动判断参数。"""
    history_len:                  int   = 8
    bootstrap_frames:             int   = 1
    max_angle_deg:                float = 16.0
    max_line_offset_px:           float = 4.0
    min_step_px:                  float = 3.0
    min_total_displacement_px:    float = 20.0
    max_missed_frames:            int   = 4
    same_direction_ratio_thresh:  float = 0.75
    bullet_min_output_streak:     int   = 5
    min_valid_step_count:         int   = 2
    min_valid_step_ratio:         float = 0.55

    # ---- P0: maintain 阶段硬约束 ----
    maintain_max_offset_px:       float = 5.0
    maintain_max_static_frames:   int   = 5
    maintain_max_bbox_std:        float = 4.5
    # ---- P0: probation 阶段线性度检查 ----
    probation_max_offset_px:      float = 4.5
    # ---- P1: ghost 跨 track 捕获 ----
    ghost_capture_enabled:        bool  = True

    # ---- P2-A: recent offset 连续超限快速 kill ----
    maintain_recent_offset_px:           float = 4.0
    maintain_recent_offset_exceed_frames: int  = 3

    # ---- P2-B: bbox EMA 慢漂移检测 ----
    maintain_bbox_ema_alpha:      float = 0.25
    maintain_bbox_ema_drift_ratio: float = 0.45

    # ---- P2-C: maintain_turn 次数和累计角度限制 ----
    maintain_max_turn_count:         int   = 2
    maintain_max_cumulative_turn_deg: float = 90.0

    # ---- P2-E / P3-1: 反弹续接最大距离（P3 缩小到 35）----
    ghost_bounce_max_dist_px:         float = 35.0

    # ---- P3-2: 反弹续接最低速度（px/ms）----
    ghost_bounce_min_speed_px_per_ms: float = 0.60

    # ---- P6-1: bbox 类终止后 same-track 重用前，要求出现新的最小位移 ----
    same_track_bbox_reuse_min_disp_px: float = 6.0

    # ---- P3-3: trigger 阶段 full_offset 倍率约束 ----
    trigger_max_full_offset_ratio:    float = 1.0

    # ---- P3-5: trigger 阶段相邻步反转比率上限 ----
    trigger_max_sign_flip_ratio:      float = 0.20

    # ---- P5-1: trigger 阶段平均速度下限（px/ms）----
    #   avg_speed = total_disp_px / track_age_ms，低于此值认为是人体/噪声慢漂移
    #   基于数据分析：真子弹 min=0.85(b2)，误判 max=0.78(b6)，折中默认值 0.60，配合 max(full,recent) 降低漏判风险
    trigger_min_avg_speed_px_per_ms:  float = 0.60

    # ---- P7-1: trigger 出生阶段的条件性收紧（主要针对 raw_id 新生轨迹）----
    trigger_raw_min_valid_steps: int = 3
    trigger_sparse_burst_max_valid_steps: int = 2
    trigger_sparse_burst_min_disp_per_step_px: float = 18.0
    trigger_sparse_burst_min_total_disp_px: float = 35.0
    trigger_sparse_burst_max_path_over_net: float = 1.03
    trigger_old_raw_age_ms: float = 120.0
    trigger_old_raw_min_valid_steps: int = 4

    # ---- P7-2: new/raw 出生后的 probation 新增运动检查 ----
    probation_new_min_extra_disp_px: float = 8.0
    probation_new_min_extra_valid_steps: int = 2

    # ---- P8: 模型轨迹主导。bbox 中心只作为观测值；model_x/model_y 作为子弹真实轨迹坐标。----
    model_track_enabled: bool = True
    model_soft_residual_px: float = 9.0      # 小残差：正常观测，较大权重校正模型
    model_hard_residual_px: float = 22.0     # 中残差：弱校正；超过则认为观测离群
    model_outlier_residual_px: float = 32.0  # 大残差：不校正，继续按预测走
    model_correction_gain: float = 0.35      # 正常观测校正权重
    model_weak_correction_gain: float = 0.10 # 可疑观测校正权重
    model_min_init_speed_px_ms: float = 0.35 # 初始化模型时的最低速度保护
    model_max_accel_px_ms2: float = 0.16     # 速度变化上限，抑制观测点把模型瞬间拉弯

    # ---- P9: 大残差观测终止保护。防止模型轨迹继续穿到人物框/连框区域。----
    model_outlier_kill_enabled: bool = True
    model_outlier_kill_frames: int = 1

    # P13: raw_id 粘背景保护。SpatterTracker 的 raw_id 有时会停留在黑色障碍物边缘，
    # 如果直接按 raw_id 绑定，会把已确认子弹轨迹拉到背景残留点上。
    raw_id_sticky_guard_enabled: bool = True
    raw_id_sticky_guard_min_speed_px_ms: float = 0.35
    raw_id_sticky_guard_static_step_px: float = 2.2
    raw_id_sticky_guard_backward_px: float = 1.0
    raw_id_sticky_guard_residual_px: float = 24.0

    # P16: confirmed bullet ID stitching after Spatter raw_id split.
    bullet_id_stitch_enabled: bool = True
    bullet_id_stitch_window_ms: float = 120.0
    bullet_id_stitch_corridor_px: float = 24.0
    bullet_id_stitch_min_dir_cos: float = 0.82
    bullet_id_stitch_min_speed_px_ms: float = 0.32
    bullet_id_stitch_max_size_ratio: float = 5.5
    bullet_id_stitch_predict_gain: float = 0.55
    bullet_id_stitch_active_max_dt_ms: float = 140.0
    bullet_id_stitch_reverse_enabled: bool = True
    bullet_id_stitch_reverse_window_ms: float = 170.0
    bullet_id_stitch_reverse_corridor_px: float = 26.0
    bullet_id_stitch_reverse_min_dir_cos: float = 0.82
    bullet_id_stitch_reverse_min_local_disp_px: float = 10.0
    bullet_id_stitch_reverse_max_size_ratio: float = 5.8

    # P19: C++ 物理子弹反弹续接。它只把反弹前后 bullet_id 归到同一发 shot，
    # 不直接证明命中；命中仍由 hit_judge 的人体 mask 接触证据决定。
    bullet_id_bounce_link_enabled: bool = True
    bullet_id_bounce_link_window_ms: float = 260.0
    bullet_id_bounce_link_max_gap_px: float = 95.0
    bullet_id_bounce_link_adaptive_max_px: float = 190.0
    bullet_id_bounce_link_min_turn_angle_deg: float = 95.0
    bullet_id_bounce_link_min_speed_px_ms: float = 0.38
    bullet_id_bounce_link_max_size_ratio: float = 6.5
    bullet_id_bounce_link_candidate_max_points: int = 12


@dataclass
class AssociationConfig:
    """目标关联代价函数参数。"""
    max_distance_px:       float = 36.0
    direction_penalty_px:  float = 10.0
    max_size_ratio:        float = 3.0


@dataclass
class DrawConfig:
    """绘制与轨迹显示参数。"""
    draw_trajectory:        bool  = True
    history_len:            int   = 40
    pre_confirm_history_len: int  = 24
    hold_missed_frames:     int   = 2
    connect_max_gap_px:     float = 72.0
    # P7: 物理拟合绘制参数。bbox 中心只作为观测值，黄色轨迹由运动学模型生成。
    kinematic_fit_enabled:  bool  = True
    kinematic_force_straight_line: bool = True  # P9：显示/输出轨迹按分段直线，不再拟合二次曲线
    kinematic_residual_px:  float = 0.0   # 0=自动；越小越不相信偏离点
    kinematic_max_curve_px: float = 0.0   # P9 默认不允许曲线弯曲
    kinematic_max_accel_px_per_ms2: float = 0.0
    kinematic_sample_step_px: float = 7.0
    draw_after_terminate_hold_ms: int = 0       # P9 默认关闭终止后飞行残留
    extend_tail_after_terminate: bool = False   # P9 默认关闭终止后外推尾迹


# ---------------------------------------------------------------------------
# 追踪状态
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    stable_track_id: int
    points:           Deque[Tuple[float, float, float, float, int]] = field(default_factory=deque)
    pre_confirm_points: Deque[Tuple[float, float, int]]             = field(default_factory=deque)
    display_points:   Deque[Tuple[float, float, int]]               = field(default_factory=deque)
    miss_count:       int   = 0
    bullet_id:        Optional[int] = None
    bullet_active:    bool  = False
    last_bullet_id:   Optional[int] = None
    confirmed_once:   bool  = False
    last_cluster:     Optional[dict] = None
    keep_streak:      int   = 0
    static_fail_count: int  = 0
    current_raw_id:   Optional[int] = None
    raw_id_history:   Deque[int] = field(default_factory=lambda: deque(maxlen=4))
    first_seen_ts:    int   = -1
    bullet_assign_ts: int   = -1
    probation_until_ts: int = -1
    probation_passed: bool  = False
    probation_fail_count: int = 0
    probation_ref_pos: Optional[Tuple[float, float]] = None
    probation_start_point_count: int = 0
    probation_start_ts: int = -1
    birth_assign_kind: str = ''
    display_bullet_id: Optional[int] = None
    # P6: 终止后短暂保留显示轨迹，避免撞击尾段在 25fps 画面里一闪而过
    hold_display_bullet_id: Optional[int] = None
    hold_display_points: Deque[Tuple[float, float, int]] = field(default_factory=deque)
    draw_hold_until_ts: int = -1
    last_terminated_ts:  int   = -1
    last_terminated_pos: Optional[Tuple[float, float]] = None
    last_terminated_dir: Optional[Tuple[float, float]] = None
    last_terminated_reason: str = ''
    segment_index: int = 0
    segment_dir:   Optional[Tuple[float, float]] = None
    # P0: bbox 滑窗 std
    recent_widths:  Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_heights: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    # P2-A: recent offset 连续超限计数
    recent_offset_exceed_count: int = 0
    # P2-B: bbox EMA（-1 表示未初始化）
    bbox_ema_w: float = -1.0
    bbox_ema_h: float = -1.0
    # P2-C: 转向统计
    turn_count:           int   = 0
    cumulative_turn_deg:  float = 0.0
    # P5-3: ghost / reuse 后的短暂 turn warmup，避免刚续接就被 overturn kill
    segment_turn_warmup_until_ts: int = -1
    # P6-1: 仅用于 same-track bbox 终止后的重用判定，记录终止时刻后是否已有新位移
    bbox_reuse_min_disp_px: float = 0.0

    # P8: 子弹运动模型状态。obs_points 是检测框中心观测；model_points 是判定/发送/绘制使用的真实轨迹。
    obs_points: Deque[Tuple[float, float, int]] = field(default_factory=lambda: deque(maxlen=48))
    model_points: Deque[Tuple[float, float, int]] = field(default_factory=lambda: deque(maxlen=80))
    model_ready: bool = False
    model_x: float = 0.0
    model_y: float = 0.0
    model_vx: float = 0.0       # px/ms
    model_vy: float = 0.0       # px/ms
    model_last_ts: int = -1
    model_residual_px: float = -1.0
    model_pred_x: float = 0.0
    model_pred_y: float = 0.0
    model_obs_used: bool = False
    model_outlier: bool = False
    model_outlier_count: int = 0
    model_update_mode: str = 'uninitialized'
    model_speed_px_ms: float = 0.0
    impact_candidate: bool = False
    impact_x: float = 0.0
    impact_y: float = 0.0
    # P10-2: model_outlier 不再第一帧直接 kill，而是进入短暂 impact/occlusion 状态。
    # 取值：'' / 'impact_candidate' / 'occlusion_predict'
    impact_state: str = ''
    impact_start_ts: int = -1
    impact_outlier_frames: int = 0
    impact_last_ts: int = -1
    # P10-4: recent segment pool 去重辅助，避免同一段反复入池。
    segment_pool_last_push_ts: int = -1
    # P10-5: 确认后回填可信候选轨迹。只在 bullet_id 已经分配后，用于显示/事件历史补前半段。
    confirmed_backfill_last_emit_ts: int = -1
    confirmed_backfill_points: Deque[Tuple[float, float, int]] = field(default_factory=lambda: deque(maxlen=32))
    # P10-6: post-outlier rearm。一个 raw/stable track 被 model_outlier/probation 撤销后，
    # 后续如果再次出现高质量短窗，允许在同一 raw 内重新开 internal segment。
    post_outlier_rearm_count: int = 0
    post_outlier_rearm_last_ts: int = -1
    post_outlier_rearm_last_window: int = 0

    # P12: 子弹穿越/撞击人体区域后的所有权继承诊断。
    # p12_owner_mode: '' / 'owner_forward' / 'owner_bounce' / 'owner_residual_guard'
    p12_owner_mode: str = ''
    p12_owner_bullet_id: int = -1
    p12_inherited_segment_index: int = -1
    p12_owner_reason: str = ''
    # P12-Fix: owner 附近的新段必须先自证稳定，不能 1~3 点就继承/显示。
    p12_owner_pending: bool = False
    p12_owner_pending_mode: str = ''
    p12_owner_pending_since_ts: int = -1
    p12_polluted_tail: bool = False


# ---------------------------------------------------------------------------
# 主过滤器
# ---------------------------------------------------------------------------

class LinearMotionFilter:
    """
    直线运动过滤器（P2 修订版）。

    用法示例::

        filt = LinearMotionFilter(
            track=TrackConfig(history_len=10, min_total_displacement_px=25.0),
            assoc=AssociationConfig(max_distance_px=40.0),
            draw=DrawConfig(draw_trajectory=True),
        )
    """

    def __init__(
        self,
        track: TrackConfig           = None,
        assoc: AssociationConfig     = None,
        draw:  DrawConfig            = None,
    ) -> None:
        self._tc = track if track is not None else TrackConfig()
        self._ac = assoc if assoc is not None else AssociationConfig()
        self._dc = draw  if draw  is not None else DrawConfig()

        tc = self._tc
        ac = self._ac
        dc = self._dc

        self.draw_trajectory         = dc.draw_trajectory
        self.draw_connect_max_gap_px = dc.connect_max_gap_px
        self.draw_hold_missed_frames = max(0, dc.hold_missed_frames)
        self.draw_history_len        = max(int(dc.history_len), tc.history_len)
        self.pre_confirm_history_len = max(int(dc.pre_confirm_history_len), tc.history_len)

        self.history_len                 = int(tc.history_len)
        self.bootstrap_frames            = int(tc.bootstrap_frames)
        self.max_angle_deg               = float(tc.max_angle_deg)
        self.max_line_offset_px          = float(tc.max_line_offset_px)
        self.min_step_px                 = float(tc.min_step_px)
        self.min_total_displacement_px   = float(tc.min_total_displacement_px)
        self.max_missed_frames           = int(tc.max_missed_frames)
        self.same_direction_ratio_thresh = float(tc.same_direction_ratio_thresh)
        self.bullet_min_output_streak    = int(tc.bullet_min_output_streak)
        self.min_valid_step_count        = int(tc.min_valid_step_count)
        self.min_valid_step_ratio        = float(tc.min_valid_step_ratio)

        self.maintain_max_offset_px      = float(tc.maintain_max_offset_px)
        self.maintain_max_static_frames  = int(tc.maintain_max_static_frames)
        self.maintain_max_bbox_std       = float(tc.maintain_max_bbox_std)
        self.probation_max_offset_px     = float(tc.probation_max_offset_px)
        self.ghost_capture_enabled       = bool(tc.ghost_capture_enabled)

        # P2-A
        self.maintain_recent_offset_px            = float(tc.maintain_recent_offset_px)
        self.maintain_recent_offset_exceed_frames = int(tc.maintain_recent_offset_exceed_frames)
        # P2-B
        self.maintain_bbox_ema_alpha       = float(tc.maintain_bbox_ema_alpha)
        self.maintain_bbox_ema_drift_ratio = float(tc.maintain_bbox_ema_drift_ratio)
        # P2-C
        self.maintain_max_turn_count          = int(tc.maintain_max_turn_count)
        self.maintain_max_cumulative_turn_deg = float(tc.maintain_max_cumulative_turn_deg)
        # P2-E / P3-1
        self.ghost_bounce_max_dist_px = float(tc.ghost_bounce_max_dist_px)
        # P3-2 / P6-1
        self.ghost_bounce_min_speed_px_per_ms = float(tc.ghost_bounce_min_speed_px_per_ms)
        self.same_track_bbox_reuse_min_disp_px = float(tc.same_track_bbox_reuse_min_disp_px)
        # P3-3
        self.trigger_max_full_offset_ratio = float(tc.trigger_max_full_offset_ratio)
        # P3-5
        self.trigger_max_sign_flip_ratio   = float(tc.trigger_max_sign_flip_ratio)
        # P5-1
        self.trigger_min_avg_speed_px_per_ms = float(tc.trigger_min_avg_speed_px_per_ms)
        # P7-1
        self.trigger_raw_min_valid_steps = int(tc.trigger_raw_min_valid_steps)
        self.trigger_sparse_burst_max_valid_steps = int(tc.trigger_sparse_burst_max_valid_steps)
        self.trigger_sparse_burst_min_disp_per_step_px = float(tc.trigger_sparse_burst_min_disp_per_step_px)
        self.trigger_sparse_burst_min_total_disp_px = float(tc.trigger_sparse_burst_min_total_disp_px)
        self.trigger_sparse_burst_max_path_over_net = float(tc.trigger_sparse_burst_max_path_over_net)
        self.trigger_old_raw_age_ms = float(tc.trigger_old_raw_age_ms)
        self.trigger_old_raw_min_valid_steps = int(tc.trigger_old_raw_min_valid_steps)
        # P7-2
        self.probation_new_min_extra_disp_px = float(tc.probation_new_min_extra_disp_px)
        self.probation_new_min_extra_valid_steps = int(tc.probation_new_min_extra_valid_steps)

        # P8: 模型轨迹参数。模型输出会写入 model_x/model_y，并用于轨迹绘制和 BulletEvent 坐标。
        self.model_track_enabled = bool(getattr(tc, 'model_track_enabled', True))
        self.model_soft_residual_px = float(getattr(tc, 'model_soft_residual_px', 9.0))
        self.model_hard_residual_px = float(getattr(tc, 'model_hard_residual_px', 22.0))
        self.model_outlier_residual_px = float(getattr(tc, 'model_outlier_residual_px', 32.0))
        self.model_correction_gain = float(getattr(tc, 'model_correction_gain', 0.35))
        self.model_weak_correction_gain = float(getattr(tc, 'model_weak_correction_gain', 0.0))
        self.model_min_init_speed_px_ms = float(getattr(tc, 'model_min_init_speed_px_ms', 0.35))
        self.model_max_accel_px_ms2 = float(getattr(tc, 'model_max_accel_px_ms2', 0.10))
        self.model_outlier_kill_enabled = bool(getattr(tc, 'model_outlier_kill_enabled', True))
        self.model_outlier_kill_frames = max(1, int(getattr(tc, 'model_outlier_kill_frames', 1)))
        # P13: raw_id 粘背景保护。只影响 raw_id 直连，不影响后续 assoc/ghost/rearm。
        self.raw_id_sticky_guard_enabled = bool(getattr(tc, 'raw_id_sticky_guard_enabled', True))
        self.raw_id_sticky_guard_min_speed_px_ms = float(getattr(tc, 'raw_id_sticky_guard_min_speed_px_ms', 0.35))
        self.raw_id_sticky_guard_static_step_px = float(getattr(tc, 'raw_id_sticky_guard_static_step_px', 2.2))
        self.raw_id_sticky_guard_backward_px = float(getattr(tc, 'raw_id_sticky_guard_backward_px', 1.0))
        self.raw_id_sticky_guard_residual_px = float(getattr(tc, 'raw_id_sticky_guard_residual_px', 24.0))
        # P10-2: 大残差先进入 impact/occlusion 短状态机。
        # 默认最多保留 2 个连续 outlier slice，第三个才终止在 impact 点；
        # 同时用时间上限兜底，避免预测线长期穿过人物区域。
        self._impact_outlier_grace_frames = max(2, self.model_outlier_kill_frames + 1)
        self._impact_max_hold_us = 12_000

        self.association_max_distance_px       = float(ac.max_distance_px)
        self.association_direction_penalty_px  = float(ac.direction_penalty_px)
        self.association_max_size_ratio        = float(ac.max_size_ratio)

        self._tracks:          Dict[int, TrackState] = {}
        self._raw_to_track:    Dict[int, int]        = {}
        self._next_track_id:   int = 1
        self._next_bullet_id:  int = 1
        self._displayed_bullet_ids: set = set()
        self._last_debug_rows: List[dict] = []
        self._recent_terminated: List[dict] = []
        # P10-4: 最近高置信 bullet segment 池。
        # 用于局部反弹/续接，不全局放宽 ghost 参数。只保存已经有 bullet_id 的段。
        self._recent_segments: List[dict] = []

        # P12: 子弹进入人体/遮挡/撞击污染区后的“所有权”池。
        # 只保存已经正式显示/通过确认的 bullet，避免人体误判自己成为 owner。
        # 后续新 raw/stable track 只有两种方式可以继承 owner：
        #   1) owner_forward：沿旧方向穿越/遮挡后继续飞；
        #   2) owner_bounce ：从 impact 点附近快速离开，允许方向偏转/反弹。
        self._occlusion_owners: List[dict] = []
        # P12-Fix: owner 只给“候选继承资格”，不允许人体残留 1~3 点直接继承/显示。
        # 因此 guard 范围稍大，但 capture 门槛明显更严。
        self._p12_owner_window_us = 240_000
        self._p12_owner_guard_us = 220_000
        self._p12_forward_corridor_px = max(28.0, self.max_line_offset_px * 7.0)
        self._p12_forward_min_project_px = max(14.0, self.min_total_displacement_px * 0.55)
        self._p12_forward_min_dir_cos = 0.62
        self._p12_bounce_birth_px = max(64.0, self.ghost_bounce_max_dist_px * 1.8)
        self._p12_bounce_adaptive_px = 130.0
        self._p12_bounce_min_leave_px = max(18.0, self.min_total_displacement_px * 0.85)
        self._p12_bounce_min_speed_px_ms = max(0.55, self.ghost_bounce_min_speed_px_per_ms * 0.85)
        self._p12_residual_guard_px = max(110.0, self.ghost_bounce_max_dist_px * 3.0)
        self._p12_commit_min_points = 4
        self._p12_commit_min_valid_steps = 3
        self._p12_commit_max_path_over_net = 1.55
        self._p12_commit_max_offset_ratio = 1.65
        self._p12_commit_min_dir_ratio = 0.78
        self._p12_commit_min_same_sign = 0.78
        self._p12_commit_max_flip = 0.30

        # P10-4-FIX: segment_bounce 内部参数默认值。上一版漏初始化这些字段，
        # 会在 update() 中触发 AttributeError。这里全部设为内部固定默认值，不需要改 JSON。
        self._recent_segment_pool_window_us = 220_000
        self._segment_bounce_dt_us = 200_000
        self._segment_bounce_max_dist_px = max(70.0, float(self.ghost_bounce_max_dist_px) * 2.0)
        self._segment_bounce_adaptive_max_px = 120.0
        self._segment_bounce_min_speed_px_ms = max(0.35, float(self.ghost_bounce_min_speed_px_per_ms) * 0.45)
        self._segment_bounce_max_size_ratio = max(4.0, float(self.association_max_size_ratio) * 1.50)
        self._segment_bounce_candidate_max_points = 6

        # P10-5: confirmed backfill。确认 bullet 后，只回填已检测到且短窗像子弹的候选点，
        # 不提前显示、不外推未来尾迹、不恢复终止残留。
        self._confirmed_backfill_enabled = True
        self._confirmed_backfill_max_points = 16
        self._confirmed_backfill_max_age_us = 90_000
        self._confirmed_backfill_min_points = 4
        self._confirmed_backfill_min_speed_px_ms = max(0.45, self.trigger_min_avg_speed_px_per_ms * 0.75)
        self._confirmed_backfill_max_path_over_net = 1.75
        self._confirmed_backfill_offset_ratio = 2.0
        self._confirmed_backfill_min_direction_ok = 0.70
        self._confirmed_backfill_min_same_sign = 0.70
        self._confirmed_backfill_max_sign_flip = 0.35

        # P10-6: 同一 raw/stable track 内的 post-outlier rearm。
        # 场景：raw 轨迹很长，但前段 bullet 被 model_outlier / probation_revoke 打断，
        # 后续又出现高质量直线短窗。此时不让旧 state.points 污染新段，
        # 直接基于最近 4~6 帧可信短窗重新开 internal segment。
        self._post_outlier_rearm_enabled = True
        self._post_outlier_rearm_window_us = 650_000
        self._post_outlier_rearm_min_dt_us = 3_000
        self._post_outlier_rearm_required_streak = 1
        self._post_outlier_rearm_reasons = {
            'model_outlier', 'probation_reuse_motion', 'probation_new_motion',
            'probation_offset', 'probation', 'recent_offset_exceed'
        }
        self._post_outlier_rearm_reset_points = True

        self._recent_window_points = min(max(4, self.history_len), 5)
        self._slow_window_points   = min(
            max(self._recent_window_points + 2, 6),
            max(self.history_len, 10),
        )

        self._late_trigger_max_age_us            = 150_000
        self._trigger_min_recent_valid_steps     = max(2, self.min_valid_step_count)
        self._trigger_min_recent_valid_ratio     = max(0.65, self.min_valid_step_ratio)
        self._trigger_max_path_over_net          = 2.20
        self._trigger_max_recent_offset_px       = self.max_line_offset_px * 1.50
        self._trigger_min_dir_ratio              = max(0.72, self.same_direction_ratio_thresh * 0.90)
        self._trigger_min_same_sign              = max(0.72, self.same_direction_ratio_thresh * 0.90)

        self._probation_window_us           = 35_000
        self._probation_min_elapsed_us      = 8_000
        self._probation_continue_disp_px    = max(8.0, self.min_total_displacement_px * 0.45)
        self._probation_recent_valid_steps  = 1
        self._probation_max_path_over_net   = 2.80
        self._probation_fail_grace          = 3

        # P10-3: rearm / ghost / same-track 出生候选的“反弹候选 probation”。
        # 目的：raw87 / raw146 / raw183 这类真实子弹段已经能触发，但容易因
        # probation_reuse_motion 在 1~2 帧内被撤销；这里只对带上下文的候选放宽，
        # 不全局降低 trigger 门槛。
        self._reuse_probation_birth_kinds = {
            'ballistic_rearm', 'post_outlier_rearm', 'same_track', 'ghost', 'ghost_capture', 'ghost_capture_bounce', 'segment_bounce', 'segment_forward'
        }
        self._reuse_probation_grace_us = 18_000
        self._reuse_probation_min_disp_px = max(6.0, self.min_total_displacement_px * 0.30)
        self._reuse_probation_min_speed_px_ms = max(0.40, self.trigger_min_avg_speed_px_per_ms * 0.65)
        self._reuse_probation_max_path_over_net = 2.80
        self._reuse_probation_max_offset_px = max(self.probation_max_offset_px * 1.35, self.max_line_offset_px * 1.80)
        self._reuse_probation_min_dir_ratio = 0.55
        self._reuse_probation_min_same_sign = 0.55
        self._reuse_probation_max_flip_ratio = 0.55
        self._reuse_probation_max_bbox_area = 14_400.0
        self._reuse_probation_max_bbox_long_side = 160.0

        self._terminate_miss_frames       = max(3, self.max_missed_frames)
        self._recent_static_disp_px       = max(0.8, self.min_step_px * 0.25)
        self._long_static_disp_px         = max(2.0, self.min_step_px * 0.70)
        self._stationary_terminate_frames = int(self.maintain_max_static_frames)
        self._maintain_size_ratio         = max(3.5, self.association_max_size_ratio * 1.8)
        self._keep_tail_points            = max(4, min(6, self.history_len))
        self._same_track_reactivate_window_us  = 100_000    # P3-1: 1_200_000 → 100ms
        self._ghost_reactivate_window_us       = 200_000    # P3-1: 3_000_000 → 200ms
        self._ghost_base_distance_px           = max(200.0, self.association_max_distance_px * 3.5)
        self._ghost_speed_px_per_us            = 0.00055
        self._segment_turn_cos                 = math.cos(math.radians(45.0))

        # P2-F: 疑似续接豁免 late_trigger 时用的时空窗口
        self._late_trigger_exempt_time_us   = 150_000   # P3-1: 400ms → 150ms
        self._late_trigger_exempt_fwd_px    = 60.0      # 直飞分支：离前向预测点的距离门限
        self._late_trigger_exempt_bounce_px = 30.0      # P3-1: 固定 30px，不再随 bounce_max 缩放
        self._segment_turn_warmup_us        = 18_000    # P5-3: reuse/ghost 后 turn 预热窗口，避免刚续接即 overturn

        # P10-1: recent ballistic rearm（低风险上线第一步）。
        # 只在 late_trigger_veto 原本会拒绝“老 raw_id”时启用；
        # 在线逻辑只检查当前 state.points 的最后 4~6 个后缀窗口，
        # 与离线 --suffix-only 回测保持一致，不枚举未来窗口。
        self._ballistic_rearm_enabled = True
        self._ballistic_rearm_window_min = 4
        self._ballistic_rearm_window_max = 6
        self._ballistic_rearm_disp_ratio = 0.80
        self._ballistic_rearm_min_valid_steps = 3
        self._ballistic_rearm_max_path_over_net = 1.45
        self._ballistic_rearm_offset_ratio = 1.80
        self._ballistic_rearm_min_direction_ok_ratio = 0.70
        self._ballistic_rearm_min_same_sign_ratio = 0.70
        self._ballistic_rearm_max_sign_flip_ratio = 0.45
        # bbox 保护使用 median + last，避免单帧撞击飞溅放大导致真子弹被误拒。
        self._ballistic_rearm_max_bbox_area = 9600.0
        self._ballistic_rearm_max_bbox_long_side = 120.0
        # ballistic_rearm 只是逃生通道，仍需要连续触发至少 2 帧后才分配 bullet_id。
        self._ballistic_rearm_required_streak = 2
        # P6: 撞击/续接尾段通常只有 1~3 个算法 slice，25fps OBS 下容易看不见，绘制端短暂 hold。
        # P9: 默认关闭终止后飞行残留/尾迹。需要调试时可通过 JSON 打开。
        self._draw_after_terminate_hold_us = max(0, int(getattr(dc, 'draw_after_terminate_hold_ms', 0))) * 1000
        self._draw_extend_tail_after_terminate = bool(getattr(dc, 'extend_tail_after_terminate', False))
        self._draw_spike_filter_enabled = True
        self._draw_spike_perp_px = max(12.0, self.max_line_offset_px * 3.0)
        self._draw_spike_min_leg_px = max(6.0, self.min_step_px * 2.0)

        # P7: 画面轨迹不再逐点连接 bbox 中心，而是对 bbox 中心观测做鲁棒运动学拟合。
        #     x(t), y(t) 采用一次/二次多项式，等价于“匀速/轻微加速度”模型。
        #     bbox 中心只作为观测值；明显偏离运动模型的点会被当作噪声降权/剔除。
        self._draw_kinematic_fit_enabled = bool(getattr(dc, 'kinematic_fit_enabled', True))
        self._draw_kinematic_force_straight_line = bool(getattr(dc, 'kinematic_force_straight_line', True))
        self._draw_kinematic_min_points = 4
        self._draw_kinematic_quad_min_points = 7
        residual_cfg = float(getattr(dc, 'kinematic_residual_px', 0.0))
        self._draw_kinematic_base_residual_px = (
            residual_cfg if residual_cfg > 0 else max(10.0, self.max_line_offset_px * 2.8)
        )
        self._draw_kinematic_max_curve_px = float(getattr(dc, 'kinematic_max_curve_px', 0.0))
        self._draw_kinematic_max_accel_px_per_ms2 = float(getattr(dc, 'kinematic_max_accel_px_per_ms2', 0.0))
        self._draw_kinematic_sample_step_px = max(3.0, float(getattr(dc, 'kinematic_sample_step_px', 7.0)))
        self._draw_kinematic_max_samples = 72
        self._last_update_ts = -1

    # -----------------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------------

    @staticmethod
    def _cluster_to_dict(cluster) -> dict:
        raw_id = int(cluster['id'])
        x = float(cluster['x'])
        y = float(cluster['y'])
        w = float(cluster['width'])
        h = float(cluster['height'])
        return {
            'id': raw_id, 'raw_id': raw_id,
            'x': x, 'y': y, 'width': w, 'height': h,
            'cx': x + 0.5 * w, 'cy': y + 0.5 * h,
        }

    def _create_state(self) -> TrackState:
        stable_track_id = self._next_track_id
        self._next_track_id += 1
        state = TrackState(
            stable_track_id=stable_track_id,
            points=deque(maxlen=self.history_len),
            pre_confirm_points=deque(maxlen=self.pre_confirm_history_len),
            display_points=deque(maxlen=self.draw_history_len),
        )
        self._tracks[stable_track_id] = state
        return state

    @staticmethod
    def _append_point_if_new(dst: Deque, point: tuple) -> None:
        if dst and dst[-1] == point:
            return
        dst.append(point)

    def _append_pre_confirm_point(self, state: TrackState, x: float, y: float, ts: int) -> None:
        self._append_point_if_new(state.pre_confirm_points, (float(x), float(y), int(ts)))

    def _append_display_point(self, state: TrackState, x: float, y: float, ts: int) -> None:
        self._append_point_if_new(state.display_points, (float(x), float(y), int(ts)))

    def _backfill_display_from_preconfirm(self, state: TrackState) -> None:
        for x, y, ts in state.pre_confirm_points:
            self._append_display_point(state, x, y, ts)

    # -------------------------------------------------------------------
    # P10-5: 确认后可信候选轨迹回填
    # -------------------------------------------------------------------

    def _select_confirmed_backfill_points(
        self, state: TrackState, current_ts: int
    ) -> List[Tuple[float, float, float, float, int]]:
        """
        P10-5: bullet_id 分配后，回看 pre_confirm_points 里的最近历史，
        只选择已经检测到、且整体满足高速直线特征的候选段。

        注意：这不是飞行残留，也不是未来外推；只是把“确认前已经真实观测到”的可信点
        回填到 model_points/display_points/debug_rows，使画线和 approach_point 不再只从确认时刻开始。
        """
        if not self._confirmed_backfill_enabled:
            return []
        cur_ts = int(current_ts)
        # 去重并按时间排序。pre_confirm_points 只有中心点，没有 bbox；bbox 用最近观测补齐。
        hist_raw = [(float(x), float(y), int(t)) for x, y, t in list(state.pre_confirm_points) if int(t) <= cur_ts]
        if len(hist_raw) < self._confirmed_backfill_min_points:
            return []
        hist_raw.sort(key=lambda p: p[2])
        hist: List[Tuple[float, float, int]] = []
        seen_ts = set()
        for x, y, t in hist_raw:
            if t in seen_ts:
                continue
            seen_ts.add(t)
            hist.append((x, y, t))
        if len(hist) < self._confirmed_backfill_min_points:
            return []

        # 只看当前时刻前的一小段历史，避免把很久以前的静态/人体点带回来。
        hist = [p for p in hist if 0 <= cur_ts - int(p[2]) <= self._confirmed_backfill_max_age_us]
        hist = hist[-self._confirmed_backfill_max_points:]
        if len(hist) < self._confirmed_backfill_min_points:
            return []

        # 用 state.points 里的时间戳补充宽高；没有匹配时使用当前 last_cluster 宽高。
        wh_by_ts = {}
        for px, py, pw, ph, pt in list(state.points):
            wh_by_ts[int(pt)] = (max(1.0, float(pw)), max(1.0, float(ph)))
        cur_w = max(1.0, float((state.last_cluster or {}).get('width', 1.0)))
        cur_h = max(1.0, float((state.last_cluster or {}).get('height', 1.0)))

        best_start = None
        best_score = -1e9
        end_idx = len(hist) - 1
        # 从更早的 start 往当前点试探，要求 start..current 整段仍然像一条弹道。
        min_start = max(0, end_idx - self._confirmed_backfill_max_points + 1)
        for start in range(min_start, end_idx - self._confirmed_backfill_min_points + 2):
            seg = hist[start:end_idx + 1]
            if len(seg) < self._confirmed_backfill_min_points:
                continue
            pseudo = []
            for x, y, t in seg:
                w, h = wh_by_ts.get(int(t), (cur_w, cur_h))
                pseudo.append((float(x), float(y), float(w), float(h), int(t)))
            stats = self._slice_stats(pseudo)
            if stats is None:
                continue
            speed = self._estimate_speed_from_points(pseudo)
            duration_us = int(seg[-1][2]) - int(seg[0][2])
            if duration_us < 3_000:
                continue
            fail = False
            if speed < self._confirmed_backfill_min_speed_px_ms:
                fail = True
            if int(stats.get('valid_step_count', 0)) < max(2, self._confirmed_backfill_min_points - 1):
                fail = True
            if float(stats.get('path_over_net', 999.0)) > self._confirmed_backfill_max_path_over_net:
                fail = True
            if float(stats.get('max_offset', 999.0)) > self.max_line_offset_px * self._confirmed_backfill_offset_ratio:
                fail = True
            if float(stats.get('direction_ok_ratio', 0.0)) < self._confirmed_backfill_min_direction_ok:
                fail = True
            if float(stats.get('same_sign_ratio', 0.0)) < self._confirmed_backfill_min_same_sign:
                fail = True
            if float(stats.get('sign_flip_ratio', 1.0)) > self._confirmed_backfill_max_sign_flip:
                fail = True
            if fail:
                continue
            # 偏好更早、更长、速度更高、几何更干净的回填段。
            score = 0.0
            score += len(seg) * 0.20
            score += min(3.0, speed)
            score += min(2.0, float(stats.get('total_disp', 0.0)) / max(1.0, self.min_total_displacement_px))
            score += max(0.0, 2.0 - float(stats.get('path_over_net', 999.0)))
            score += max(0.0, 2.0 - float(stats.get('max_offset', 999.0)) / max(1.0, self.max_line_offset_px))
            # 同分时选更早的 start。
            if best_start is None or score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and start < best_start):
                best_start = start
                best_score = score

        if best_start is None:
            return []

        out: List[Tuple[float, float, float, float, int]] = []
        for x, y, t in hist[best_start:end_idx + 1]:
            w, h = wh_by_ts.get(int(t), (cur_w, cur_h))
            out.append((float(x), float(y), float(w), float(h), int(t)))
        return out

    def _apply_confirmed_backfill_points(
        self, state: TrackState, pts: List[Tuple[float, float, float, float, int]]
    ) -> None:
        """把 P10-5 可信候选点合入 model_points/display_points。"""
        if not pts:
            return
        # 转成 model/display 使用的 (x,y,ts)，按时间去重。
        merged = []
        seen = set()
        for x, y, _w, _h, t in pts:
            ti = int(t)
            if ti in seen:
                continue
            seen.add(ti)
            merged.append((float(x), float(y), ti))
        for x, y, t in list(state.model_points):
            ti = int(t)
            if ti in seen:
                continue
            seen.add(ti)
            merged.append((float(x), float(y), ti))
        merged.sort(key=lambda p: p[2])
        if not merged:
            return
        state.model_points = deque(merged[-max(80, self.draw_history_len):], maxlen=80)
        state.display_points = deque(merged[-self.draw_history_len:], maxlen=self.draw_history_len)
        state.confirmed_backfill_points.clear()
        for x, y, t in merged[-32:]:
            state.confirmed_backfill_points.append((float(x), float(y), int(t)))

    def _build_confirmed_backfill_debug_rows(
        self, state: TrackState, raw_id: int, stable_track_id: int,
        current_cluster: dict, pts: List[Tuple[float, float, float, float, int]], current_ts: int,
    ) -> List[dict]:
        """生成一次性虚拟 debug rows，让 effect 日志和 BulletEventExtractor 能拿到确认前 approach 点。"""
        if not pts or state.bullet_id is None:
            return []
        rows: List[dict] = []
        cur_ts = int(current_ts)
        bullet_id = int(state.bullet_id)
        disp_id = int(state.display_bullet_id) if state.display_bullet_id is not None else -1
        # 当前 bbox 兜底值
        base_w = max(1.0, float(current_cluster.get('width', 1.0)))
        base_h = max(1.0, float(current_cluster.get('height', 1.0)))
        for x, y, w, h, t in pts:
            ti = int(t)
            # 当前帧会由真实 row 输出；这里只补历史点。
            if ti >= cur_ts:
                continue
            if ti <= int(state.confirmed_backfill_last_emit_ts):
                continue
            ww = max(1.0, float(w or base_w))
            hh = max(1.0, float(h or base_h))
            rows.append({
                'timestamp': ti, 'raw_id': int(raw_id), 'stable_track_id': int(stable_track_id),
                'assign_kind': 'confirmed_backfill',
                'x': float(x - 0.5 * ww), 'y': float(y - 0.5 * hh),
                'width': float(ww), 'height': float(hh),
                'cx': float(x), 'cy': float(y),
                'obs_x': float(x), 'obs_y': float(y),
                'model_x': float(x), 'model_y': float(y),
                'pred_x': float(x), 'pred_y': float(y),
                'model_residual_px': 0.0, 'model_obs_used': 1, 'model_outlier': 0,
                'model_update_mode': 'confirmed_backfill',
                'model_speed_px_ms': float(state.model_speed_px_ms),
                'impact_candidate': int(state.impact_candidate),
                'impact_x': float(state.impact_x), 'impact_y': float(state.impact_y),
                'impact_state': str(state.impact_state),
                'impact_outlier_frames': int(state.impact_outlier_frames),
                'n_points': len(state.points), 'model_n_points': len(state.model_points), 'miss_count': int(state.miss_count),
                'keep_now': 1, 'confirmed_now': 1, 'accepted_now': 1,
                'bullet_id': bullet_id, 'display_bullet_id': disp_id,
                'just_assigned_bullet': 0, 'birth_assign_kind': state.birth_assign_kind,
                'keep_streak_after': int(state.keep_streak), 'reject_reason': 'confirmed_backfill',
                'line_ok': 1, 'direction_ok': 1, 'motion_ok': 1,
                'total_disp_px': -1.0, 'max_offset_px': -1.0, 'mean_offset_px': -1.0,
                'direction_ok_ratio': -1.0, 'same_sign_ratio': -1.0,
                'valid_step_count': -1, 'valid_step_ratio': -1.0, 'total_step_count': -1,
                'phase': 'confirmed_backfill', 'trigger_ok': 0, 'maintain_ok': 1,
                'bullet_active': 1, 'last_bullet_id': int(state.last_bullet_id) if state.last_bullet_id is not None else -1,
                'static_fail_count': int(state.static_fail_count), 'trigger_disp_thresh': float(self.min_total_displacement_px),
                'recent_total_disp_px': -1.0, 'recent_max_offset_px': -1.0, 'recent_mean_offset_px': -1.0,
                'recent_direction_ok_ratio': -1.0, 'recent_same_sign_ratio': -1.0,
                'recent_valid_step_count': -1, 'recent_valid_step_ratio': -1.0,
                'recent_geom_ok': 1, 'recent_motion_ok': 1,
                'track_age_ms': float((ti - int(state.first_seen_ts)) / 1000.0) if state.first_seen_ts >= 0 else -1.0,
                'path_over_net': -1.0, 'recent_path_over_net': -1.0,
                'reuse_source': 'confirmed_backfill',
                'recent_segment_pool_size': int(len(self._recent_segments)),
                'segment_pool_candidate': 0,
                'probation_passed': int(state.probation_passed),
                'probation_fail_count': int(state.probation_fail_count),
                'segment_index': int(state.segment_index),
                'compact_ok': 1,
                'rearm_mode': 0, 'ballistic_rearm': 0, 'ballistic_rearm_window': 0,
                'ballistic_rearm_speed_px_ms': 0.0,
                'post_outlier_rearm': 0, 'post_outlier_rearm_window': 0,
                'post_outlier_rearm_dt_us': 0, 'post_outlier_rearm_count': int(state.post_outlier_rearm_count),
                'required_trigger_streak': 0,
                'probation_reuse_soft_ok': 0, 'probation_reuse_strict_ok': 0,
                'turn_count': int(state.turn_count),
                'cumulative_turn_deg': float(state.cumulative_turn_deg),
                'recent_offset_exceed_count': int(state.recent_offset_exceed_count),
                'sign_flip_ratio': -1.0, 'recent_sign_flip_ratio': -1.0,
                # P10-5 诊断字段；主脚本旧 CSV 表头会忽略，未来更新表头后可见。
                'confirmed_backfill': 1,
            })
        if rows:
            state.confirmed_backfill_last_emit_ts = max(int(r['timestamp']) for r in rows)
        return rows

    def _trim_points(self, state: TrackState, keep_last: Optional[int] = None) -> None:
        keep_last = self._keep_tail_points if keep_last is None else int(keep_last)
        pts = list(state.points)
        state.points = deque(pts[-keep_last:], maxlen=self.history_len)

    def _predict_center(self, state: TrackState) -> Optional[Tuple[float, float]]:
        pts = list(state.points)
        if not pts:
            return None
        if len(pts) == 1:
            return float(pts[-1][0]), float(pts[-1][1])
        p0 = np.array([pts[-2][0], pts[-2][1]], dtype=np.float32)
        p1 = np.array([pts[-1][0], pts[-1][1]], dtype=np.float32)
        step = p1 - p0
        norm = float(np.linalg.norm(step))
        if norm < 1e-6:
            return float(p1[0]), float(p1[1])
        scale = min(1.6, 1.0 + 0.25 * max(0, state.miss_count))
        pred = p1 + step * scale
        return float(pred[0]), float(pred[1])

    def _track_direction(
        self, state: TrackState, window_points: Optional[int] = None
    ) -> Optional[np.ndarray]:
        pts = list(state.points)
        if window_points is not None:
            pts = pts[-int(window_points):]
        if len(pts) < 2:
            return None
        p0 = np.array([pts[0][0], pts[0][1]], dtype=np.float32)
        p1 = np.array([pts[-1][0], pts[-1][1]], dtype=np.float32)
        vec = p1 - p0
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm

    def _association_cost(self, state: TrackState, c: dict) -> Optional[float]:
        pred = self._predict_center(state)
        if pred is None:
            return None
        pred_xy = np.array(pred, dtype=np.float32)
        cur_xy  = np.array([c['cx'], c['cy']], dtype=np.float32)
        dist    = float(np.linalg.norm(cur_xy - pred_xy))

        allow_dist = self.association_max_distance_px * (1.0 + 0.35 * max(0, state.miss_count))
        if state.bullet_active:
            allow_dist *= 1.40
        if dist > allow_dist:
            return None

        size_ratio = 1.0
        last_cluster = state.last_cluster
        if last_cluster is not None:
            last_w = max(1.0, float(last_cluster['width']))
            last_h = max(1.0, float(last_cluster['height']))
            cur_w  = max(1.0, float(c['width']))
            cur_h  = max(1.0, float(c['height']))
            size_ratio = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            allowed_sr = self.association_max_size_ratio * (1.45 if state.bullet_active else 1.0)
            if size_ratio > allowed_sr:
                return None

        cost = dist + (size_ratio - 1.0) * 4.0

        direction = self._track_direction(state, window_points=min(4, max(2, len(state.points))))
        if direction is not None:
            last_pt  = np.array([state.points[-1][0], state.points[-1][1]], dtype=np.float32)
            obs_vec  = cur_xy - last_pt
            obs_norm = float(np.linalg.norm(obs_vec))
            if obs_norm >= max(1.0, self.min_step_px * 0.4):
                obs_dir   = obs_vec / max(obs_norm, 1e-6)
                cos_val   = float(np.dot(obs_dir, direction))
                penalty_scale = 0.30 if state.bullet_active else 1.0
                if (not state.bullet_active) and cos_val < -0.15:
                    cost += (1.0 - abs(max(-1.0, min(1.0, cos_val)))) * (
                        self.association_direction_penalty_px * 3.0
                    )
                else:
                    cost += (1.0 - abs(max(-1.0, min(1.0, cos_val)))) * (
                        self.association_direction_penalty_px * penalty_scale
                    )
        return cost

    def _direct_raw_id_association_ok(self, state: TrackState, c: dict, ts: int) -> bool:
        """P13: Spatter raw_id 粘背景保护。

        SpatterTracker 在场地/黑色障碍物交界处可能把同一个 raw_id 留在背景边缘。
        旧逻辑会先按 raw_id 直接绑定，导致已确认子弹轨迹被静止背景点拉断。
        这里只拦截“已确认/曾确认子弹”的 raw_id 直连；被拦截的 cluster 仍可走
        空间 assoc/新 track，因此不会全局丢候选。
        """
        if not self.raw_id_sticky_guard_enabled:
            return True
        if not (state.bullet_active or state.confirmed_once or state.bullet_id is not None or state.display_bullet_id is not None):
            return True
        if len(state.points) < 2:
            return True

        cx = float(c.get('cx', 0.0)); cy = float(c.get('cy', 0.0))
        # 对已确认子弹，优先用模型预测点；模型不可用时才用最后两点预测。
        if getattr(state, 'model_ready', False):
            pred_x, pred_y = self._predict_model_xy(state, int(ts))
            dir_vec = None
            speed = float(math.hypot(float(state.model_vx), float(state.model_vy)))
            if speed > 1e-6:
                dir_vec = np.array([float(state.model_vx) / speed, float(state.model_vy) / speed], dtype=np.float32)
        else:
            pred = self._predict_center(state)
            if pred is None:
                return True
            pred_x, pred_y = pred
            d = self._track_direction(state, window_points=min(4, max(2, len(state.points))))
            dir_vec = d
            speed = float(self._estimate_recent_speed_px_per_ms(state))

        residual = float(math.hypot(cx - float(pred_x), cy - float(pred_y)))
        last_pt = state.points[-1]
        step_x = cx - float(last_pt[0]); step_y = cy - float(last_pt[1])
        step_len = float(math.hypot(step_x, step_y))

        # 1) 已确认高速子弹不应在同一个背景点上原地停住。
        if speed >= self.raw_id_sticky_guard_min_speed_px_ms and step_len <= self.raw_id_sticky_guard_static_step_px:
            return False

        # 2) 不允许 raw_id 直连把点明显拉到预测轨迹之外；后续 assoc/ghost 会负责找新 raw_id 续接。
        residual_limit = max(8.0, float(self.raw_id_sticky_guard_residual_px))
        if residual > residual_limit:
            return False

        # 3) 若有明确运动方向，禁止明显反向/不前进的 raw_id 背景残留继续绑定旧弹道。
        if dir_vec is not None and speed >= self.raw_id_sticky_guard_min_speed_px_ms:
            fwd = float(step_x * float(dir_vec[0]) + step_y * float(dir_vec[1]))
            if fwd < -abs(self.raw_id_sticky_guard_backward_px):
                return False
            if step_len <= max(self.raw_id_sticky_guard_static_step_px, self.min_step_px * 0.75) and fwd < self.min_step_px * 0.20:
                return False

        # 4) raw_id 直连以前绕过了 bbox size ratio，这里补上，避免障碍物边缘大框粘回弹道。
        last_cluster = state.last_cluster
        if last_cluster is not None:
            last_w = max(1.0, float(last_cluster.get('width', 1.0)))
            last_h = max(1.0, float(last_cluster.get('height', 1.0)))
            cur_w = max(1.0, float(c.get('width', 1.0)))
            cur_h = max(1.0, float(c.get('height', 1.0)))
            sr = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            if sr > self.association_max_size_ratio * 1.45:
                return False

        return True

    def _assign_clusters(
        self, clusters: List[dict]
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        assignments:      Dict[int, int] = {}
        assign_kind:      Dict[int, str] = {}
        matched_track_ids: set = set()
        pending_clusters: List[dict] = []

        for c in clusters:
            raw_id          = int(c['raw_id'])
            stable_track_id = self._raw_to_track.get(raw_id)
            state = self._tracks.get(stable_track_id) if stable_track_id is not None else None
            if state is not None and stable_track_id not in matched_track_ids and self._direct_raw_id_association_ok(state, c, self._last_update_ts):
                assignments[raw_id] = stable_track_id
                assign_kind[raw_id] = 'raw_id'
                matched_track_ids.add(stable_track_id)
            else:
                # P14: raw_id 粘背景导致同一颗子弹换 raw_id 时，要先把旧 bullet 的
                # 最后可信模型段放入 recent_segment 池。否则真实子弹后续产生的新 raw_id
                # 会被当成全新 bullet_id，形成“一条弹道多个 bullet_id”。
                if state is not None and stable_track_id not in matched_track_ids:
                    if (state.bullet_active or state.confirmed_once or state.bullet_id is not None or state.display_bullet_id is not None):
                        self._push_recent_segment(state, 'raw_id_sticky_handoff', int(self._last_update_ts))
                        # 也放入 owner 池，让黑色障碍物交界处的新 raw_id 可以走更严格的 forward/bounce 继承。
                        self._push_occlusion_owner(state, 'raw_id_sticky_handoff', int(self._last_update_ts))
                # P13: raw_id 粘背景时不要直接绑定旧弹道，交给空间 assoc 或新 track 处理。
                pending_clusters.append(c)

        candidate_pairs: List[Tuple[float, int, int]] = []
        for c in pending_clusters:
            raw_id = int(c['raw_id'])
            for stable_track_id, state in self._tracks.items():
                if stable_track_id in matched_track_ids:
                    continue
                cost = self._association_cost(state, c)
                if cost is not None:
                    candidate_pairs.append((float(cost), raw_id, stable_track_id))
        candidate_pairs.sort(key=lambda x: x[0])

        matched_raw_ids: set = set()
        for _, raw_id, stable_track_id in candidate_pairs:
            if raw_id in matched_raw_ids or stable_track_id in matched_track_ids:
                continue
            assignments[raw_id] = stable_track_id
            assign_kind[raw_id] = 'assoc'
            matched_raw_ids.add(raw_id)
            matched_track_ids.add(stable_track_id)

        for c in clusters:
            raw_id = int(c['raw_id'])
            if raw_id in assignments:
                continue
            state = self._create_state()
            assignments[raw_id] = state.stable_track_id
            assign_kind[raw_id] = 'new'
            matched_track_ids.add(state.stable_track_id)

        return assignments, assign_kind

    def _rebuild_raw_map(self) -> None:
        mapping: Dict[int, int] = {}
        for stable_track_id, state in self._tracks.items():
            if state.current_raw_id is not None:
                mapping[int(state.current_raw_id)] = stable_track_id
            for raw_id in state.raw_id_history:
                mapping[int(raw_id)] = stable_track_id
        self._raw_to_track = mapping

    def _purge_recent_terminated(self, now_ts: int) -> None:
        self._recent_terminated = [
            item for item in self._recent_terminated
            if now_ts - int(item.get('ts', now_ts)) <= self._ghost_reactivate_window_us
        ]

    # ------------------------------------------------------------------
    # P10-4: recent segment pool
    # ------------------------------------------------------------------
    def _purge_recent_segments(self, now_ts: int) -> None:
        self._recent_segments = [
            item for item in self._recent_segments
            if int(now_ts) - int(item.get('ts', now_ts)) <= self._recent_segment_pool_window_us
        ]

    # ------------------------------------------------------------------
    # P12: 子弹人体区域穿越 / 撞击 / 反弹的所有权管理
    # ------------------------------------------------------------------
    def _purge_occlusion_owners(self, now_ts: int) -> None:
        now_i = int(now_ts)
        self._occlusion_owners = [
            item for item in self._occlusion_owners
            if now_i - int(item.get('ts', now_i)) <= self._p12_owner_window_us
        ]

    def _is_confirmed_owner_source(self, state: TrackState) -> bool:
        """只有已经正式显示/确认过的 bullet 才能成为 P12 owner。"""
        if state.bullet_id is None:
            return False
        bid = int(state.bullet_id)
        return bool(
            state.display_bullet_id is not None
            or bid in self._displayed_bullet_ids
            or state.probation_passed
        )

    def _push_occlusion_owner(self, state: TrackState, reason: str, ts_override: Optional[int] = None) -> None:
        """把已确认 bullet 的最后可靠撞击/遮挡点放入 P12 owner 池。

        与 recent_segment 不同，这个池只服务于“物理同一颗子弹”的所有权继承，
        并且只接受已经正式显示/确认过的 bullet，避免人体假段成为 owner。
        """
        if state.bullet_id is None or not self._is_confirmed_owner_source(state):
            return
        ts_i = int(ts_override if ts_override is not None else (state.last_cluster.get('ts', 0) if state.last_cluster else -1))
        if ts_i < 0:
            return
        bid = int(state.bullet_id)

        # 选择 owner 位置：优先 impact 点，其次模型点，最后观测点。
        if state.impact_state and (state.impact_x != 0.0 or state.impact_y != 0.0):
            pos = (float(state.impact_x), float(state.impact_y))
        elif state.model_ready:
            pos = (float(state.model_x), float(state.model_y))
        elif state.last_cluster is not None:
            pos = (float(state.last_cluster.get('cx', 0.0)), float(state.last_cluster.get('cy', 0.0)))
        else:
            return

        direction = self._model_direction(state)
        if direction is None and state.segment_dir is not None:
            direction = np.array(state.segment_dir, dtype=np.float32)
        if direction is None and state.last_terminated_dir is not None:
            direction = np.array(state.last_terminated_dir, dtype=np.float32)
        dir_tuple = None
        if direction is not None:
            direction = np.array(direction, dtype=np.float32)
            dn = float(np.linalg.norm(direction))
            if dn > 1e-6:
                direction = direction / dn
                dir_tuple = (float(direction[0]), float(direction[1]))

        width = float(state.last_cluster.get('width', 1.0)) if state.last_cluster is not None else 1.0
        height = float(state.last_cluster.get('height', 1.0)) if state.last_cluster is not None else 1.0
        item = {
            'bullet_id': bid,
            'segment_index': int(state.segment_index),
            'ts': ts_i,
            'pos': pos,
            'dir': dir_tuple,
            'width': max(1.0, width),
            'height': max(1.0, height),
            'stable_track_id': int(state.stable_track_id),
            'raw_id': int(state.current_raw_id) if state.current_raw_id is not None else -1,
            'reason': str(reason),
            'recent_speed_px_per_ms': float(state.model_speed_px_ms if state.model_ready else self._estimate_recent_speed_px_per_ms(state)),
            'display_committed': bool(state.display_bullet_id is not None or bid in self._displayed_bullet_ids),
        }

        # 同一 bullet/track 在相邻 slice 连续 outlier 时更新 owner，而不是堆积重复项。
        for old in reversed(self._occlusion_owners):
            if (int(old.get('bullet_id', -1)) == bid
                    and int(old.get('stable_track_id', -2)) == int(state.stable_track_id)
                    and abs(ts_i - int(old.get('ts', ts_i))) <= 6_000):
                old.update(item)
                self._purge_occlusion_owners(ts_i)
                return
        self._occlusion_owners.append(item)
        self._purge_occlusion_owners(ts_i)

    def _candidate_motion_summary_for_owner(self, state: TrackState) -> dict:
        """P12-Fix：给 owner 续接使用的局部运动质量摘要。

        这里故意比 trigger/early-display 更严格：owner 附近人体残留也可能在
        1~3 个点内看起来像直线，所以只有当前新段自己形成 4+ 点、3+ 有效步、
        低 path_over_net、低 offset、方向稳定后，才允许真正继承旧 bullet_id。
        """
        ok, stats, info = self._select_ballistic_rearm_suffix_stats(state)
        pts = list(state.points)
        local_points = len(pts)
        speed = float(info.get('speed_px_ms', 0.0) or self._estimate_recent_speed_px_per_ms(state)) if ok else float(self._estimate_recent_speed_px_per_ms(state))
        disp = 0.0
        direction = None
        if stats is not None and stats.get('direction') is not None:
            direction = stats.get('direction')
        if local_points >= 2:
            p0 = np.array([pts[0][0], pts[0][1]], dtype=np.float32)
            p1 = np.array([pts[-1][0], pts[-1][1]], dtype=np.float32)
            vec = p1 - p0
            disp = float(np.linalg.norm(vec))
            if direction is None and disp > 1e-6:
                direction = vec / disp

        valid_steps = int(stats.get('valid_step_count', 0)) if stats is not None else 0
        path_over_net = float(stats.get('path_over_net', 999.0)) if stats is not None else 999.0
        max_offset = float(stats.get('max_offset', 999.0)) if stats is not None else 999.0
        dir_ratio = float(stats.get('direction_ok_ratio', 0.0)) if stats is not None else 0.0
        same_sign = float(stats.get('same_sign_ratio', 0.0)) if stats is not None else 0.0
        flip_ratio = float(stats.get('sign_flip_ratio', 1.0)) if stats is not None else 1.0
        total_disp = float(stats.get('total_disp', disp)) if stats is not None else disp
        disp = max(disp, total_disp)

        strict_ok = bool(
            stats is not None
            and local_points >= self._p12_commit_min_points
            and valid_steps >= self._p12_commit_min_valid_steps
            and disp >= max(18.0, self.min_total_displacement_px * 0.85)
            and speed >= max(0.48, self.trigger_min_avg_speed_px_per_ms * 0.80)
            and path_over_net <= self._p12_commit_max_path_over_net
            and max_offset <= self.max_line_offset_px * self._p12_commit_max_offset_ratio
            and dir_ratio >= self._p12_commit_min_dir_ratio
            and same_sign >= self._p12_commit_min_same_sign
            and flip_ratio <= self._p12_commit_max_flip
        )
        return {
            'ok': bool(strict_ok),
            'stats': stats,
            'points': int(local_points),
            'valid_steps': int(valid_steps),
            'speed': float(speed),
            'disp': float(disp),
            'path_over_net': float(path_over_net),
            'max_offset': float(max_offset),
            'direction_ok_ratio': float(dir_ratio),
            'same_sign_ratio': float(same_sign),
            'sign_flip_ratio': float(flip_ratio),
            'direction': direction,
        }

    def _is_near_occlusion_owner(self, c: dict, ts: int, guard_px: Optional[float] = None) -> bool:
        """是否处在已确认子弹的 impact/occlusion 污染保护区内。"""
        ts_i = int(ts)
        self._purge_occlusion_owners(ts_i)
        if not self._occlusion_owners:
            return False
        cur_xy = np.array([float(c.get('cx', 0.0)), float(c.get('cy', 0.0))], dtype=np.float32)
        radius = float(self._p12_residual_guard_px if guard_px is None else guard_px)
        for item in self._occlusion_owners:
            dt = ts_i - int(item.get('ts', ts_i))
            if dt <= 0 or dt > self._p12_owner_guard_us:
                continue
            prev_xy = np.array(item.get('pos', (0.0, 0.0)), dtype=np.float32)
            if float(np.linalg.norm(cur_xy - prev_xy)) <= radius:
                return True
        return False

    def _try_occlusion_owner_capture_new_track(
        self, state: TrackState, c: dict, ts: int
    ) -> Tuple[Optional[int], str]:
        """P12-Fix owner 续接。

        owner 不再直接给新轨迹 bullet_id。只有新段已经形成严格、稳定、
        干净的局部弹道后，才返回 owner_forward / owner_bounce。
        如果仍在 owner 附近但质量不足，标记 pending，交给 residual guard 静默等待/撤销。
        """
        ts_i = int(ts)
        state.p12_owner_mode = ''
        state.p12_owner_bullet_id = -1
        state.p12_inherited_segment_index = -1
        state.p12_owner_reason = ''
        state.p12_owner_pending = False
        state.p12_owner_pending_mode = ''
        if state.p12_owner_pending_since_ts < 0:
            state.p12_owner_pending_since_ts = ts_i
        self._purge_occlusion_owners(ts_i)
        if not self._occlusion_owners:
            state.p12_owner_pending_since_ts = -1
            return None, 'none'

        cur_xy = np.array([float(c['cx']), float(c['cy'])], dtype=np.float32)
        cur_w = max(1.0, float(c.get('width', 1.0)))
        cur_h = max(1.0, float(c.get('height', 1.0)))
        q = self._candidate_motion_summary_for_owner(state)
        local_ok = bool(q.get('ok', False))
        local_dir = q.get('direction')
        local_speed = float(q.get('speed', 0.0))
        local_disp = float(q.get('disp', 0.0))
        local_points = int(q.get('points', 0))

        # 质量不够时只记 pending，不继承。这样人体残留不会因为靠近 impact 点就显示。
        if not local_ok:
            if self._is_near_occlusion_owner(c, ts_i):
                state.p12_owner_pending = True
                state.p12_owner_pending_mode = 'owner_pending_quality'
            return None, 'none'

        candidates = []
        for item in self._occlusion_owners:
            bid = int(item.get('bullet_id', -1))
            if bid <= 0:
                continue
            dt = ts_i - int(item.get('ts', ts_i))
            if dt <= 0 or dt > self._p12_owner_window_us:
                continue
            prev_xy = np.array(item.get('pos', (0.0, 0.0)), dtype=np.float32)
            delta = cur_xy - prev_xy
            dist = float(np.linalg.norm(delta))
            dt_ms = dt / 1000.0
            implied_speed = dist / max(dt_ms, 1e-6)

            last_w = max(1.0, float(item.get('width', cur_w)))
            last_h = max(1.0, float(item.get('height', cur_h)))
            sr = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            if sr > min(self._segment_bounce_max_size_ratio, 4.8):
                continue

            owner_speed = float(item.get('recent_speed_px_per_ms', 0.0))
            owner_seg_idx = int(item.get('segment_index', 0))

            # A. 穿越 / 遮挡：必须在旧方向前方，且在更窄的预测走廊内。
            owner_dir = item.get('dir')
            if owner_dir is not None:
                gd = np.array(owner_dir, dtype=np.float32)
                gn = float(np.linalg.norm(gd))
                if gn > 1e-6 and dist > 1e-6:
                    gd = gd / gn
                    project = float(np.dot(delta, gd))
                    perp_vec = delta - gd * project
                    perp = float(np.linalg.norm(perp_vec))
                    corridor = self._p12_forward_corridor_px + min(18.0, max(owner_speed, local_speed) * dt_ms * 0.16)
                    local_dir_ok = False
                    if local_dir is not None:
                        ld = np.array(local_dir, dtype=np.float32)
                        ln = float(np.linalg.norm(ld))
                        if ln > 1e-6:
                            local_dir_ok = float(np.dot(ld / ln, gd)) >= self._p12_forward_min_dir_cos
                    if (project >= self._p12_forward_min_project_px
                            and perp <= corridor
                            and implied_speed >= max(0.42, self.trigger_min_avg_speed_px_per_ms * 0.70)
                            and local_dir_ok):
                        score = perp + 0.00010 * dt + max(0.0, sr - 1.0) * 8.0 - min(10.0, implied_speed * 1.0)
                        candidates.append((score, bid, 'owner_forward', owner_seg_idx, item))

            # B. 撞击 / 反弹：方向可变，但必须由“已稳定的新段”证明自己在离开 impact 点。
            adaptive_bounce = max(
                self._p12_bounce_birth_px,
                min(self._p12_bounce_adaptive_px, 16.0 + max(owner_speed, local_speed) * dt_ms * 0.90),
            )
            moving_away_ok = False
            if local_dir is not None and dist > 1e-6:
                ld = np.array(local_dir, dtype=np.float32)
                ln = float(np.linalg.norm(ld))
                if ln > 1e-6:
                    moving_away_ok = float(np.dot(ld / ln, delta / dist)) >= 0.45
            if (dist <= adaptive_bounce
                    and dist >= self._p12_bounce_min_leave_px
                    and local_disp >= self._p12_bounce_min_leave_px
                    and implied_speed >= self._p12_bounce_min_speed_px_ms
                    and moving_away_ok):
                score = dist + 0.00014 * dt + max(0.0, sr - 1.0) * 8.0 - min(12.0, max(local_speed, implied_speed) * 1.1)
                candidates.append((score, bid, 'owner_bounce', owner_seg_idx + 1, item))

        if not candidates:
            if self._is_near_occlusion_owner(c, ts_i):
                state.p12_owner_pending = True
                state.p12_owner_pending_mode = 'owner_pending_no_match'
            return None, 'none'
        candidates.sort(key=lambda x: x[0])
        _score, bid, src, seg_idx, item = candidates[0]
        state.p12_owner_mode = str(src)
        state.p12_owner_bullet_id = int(bid)
        state.p12_inherited_segment_index = int(seg_idx)
        state.p12_owner_reason = str(item.get('reason', ''))
        state.p12_owner_pending = False
        state.p12_owner_pending_mode = ''
        state.p12_owner_pending_since_ts = -1
        return int(bid), str(src)

    def _push_recent_segment(self, state: TrackState, reason: str, ts_override: Optional[int] = None) -> None:
        """P10-4: 把最近一个可信 bullet segment 放入局部续接池。

        只保存已经有 bullet_id 的段，供新 raw_id 做 segment_bounce / segment_forward。
        与 _recent_terminated 的区别：这里也会在 impact_candidate 阶段入池，
        这样反弹新 raw_id 不必等旧段彻底 deactivate 后才有候选上下文。
        """
        if state.bullet_id is None:
            return
        ts_i = int(ts_override if ts_override is not None else (state.last_cluster.get('ts', 0) if state.last_cluster else -1))
        if ts_i < 0:
            return
        # 同一 state 在相邻 3ms slice 可能反复 impact，避免池子刷屏。
        if state.segment_pool_last_push_ts >= 0 and abs(ts_i - int(state.segment_pool_last_push_ts)) <= 3_500:
            return
        direction = self._model_direction(state)
        if direction is None:
            direction = state.last_terminated_dir
        if state.impact_state and (state.impact_x != 0.0 or state.impact_y != 0.0):
            pos = (float(state.impact_x), float(state.impact_y))
        elif state.model_ready:
            pos = (float(state.model_x), float(state.model_y))
        elif state.last_cluster is not None:
            pos = (float(state.last_cluster.get('cx', 0.0)), float(state.last_cluster.get('cy', 0.0)))
        else:
            return
        width = float(state.last_cluster.get('width', 1.0)) if state.last_cluster is not None else 1.0
        height = float(state.last_cluster.get('height', 1.0)) if state.last_cluster is not None else 1.0
        item = {
            'bullet_id': int(state.bullet_id),
            'segment_index': int(state.segment_index),
            'ts': ts_i,
            'pos': pos,
            'dir': (float(direction[0]), float(direction[1])) if direction is not None else None,
            'width': max(1.0, width),
            'height': max(1.0, height),
            'stable_track_id': int(state.stable_track_id),
            'raw_id': int(state.current_raw_id) if state.current_raw_id is not None else -1,
            'reason': str(reason),
            'recent_speed_px_per_ms': float(state.model_speed_px_ms if state.model_ready else self._estimate_recent_speed_px_per_ms(state)),
        }
        self._recent_segments.append(item)
        state.segment_pool_last_push_ts = ts_i
        self._purge_recent_segments(ts_i)

    def _push_recent_terminated(self, state: TrackState, reason: str) -> None:
        if state.bullet_id is None or state.last_cluster is None:
            return
        direction = self._model_direction(state) or (None if not state.model_ready else None)
        term_x = float(state.model_x) if state.model_ready else float(state.last_cluster['cx'])
        term_y = float(state.model_y) if state.model_ready else float(state.last_cluster['cy'])
        self._recent_terminated.append({
            'bullet_id':      int(state.bullet_id),
            'ts':             int(state.last_cluster.get('ts', 0)),
            'pos':            (term_x, term_y),
            'dir':            (float(direction[0]), float(direction[1])) if direction is not None else None,
            'width':          float(state.last_cluster['width']),
            'height':         float(state.last_cluster['height']),
            'stable_track_id': int(state.stable_track_id),
            'raw_id':         int(state.current_raw_id) if state.current_raw_id is not None else -1,
            'reason':         reason,
            'recent_speed_px_per_ms': float(state.model_speed_px_ms if state.model_ready else self._estimate_recent_speed_px_per_ms(state)),
        })

    def _deactivate_bullet(self, state: TrackState, reason: str) -> None:
        if state.bullet_active and state.bullet_id is not None:
            state.last_bullet_id       = int(state.bullet_id)
            state.last_terminated_ts   = int(state.last_cluster.get('ts', 0)) if state.last_cluster else -1
            if state.model_ready:
                state.last_terminated_pos = (float(state.model_x), float(state.model_y))
            elif state.last_cluster is not None:
                state.last_terminated_pos = (
                    float(state.last_cluster['cx']), float(state.last_cluster['cy'])
                )
            direction = self._model_direction(state)
            state.last_terminated_dir    = (float(direction[0]), float(direction[1])) if direction is not None else None
            state.last_terminated_reason = reason
            if reason in {'model_outlier', 'recent_offset_exceed', 'bbox_unstable', 'bbox_ema_drift'}:
                self._push_occlusion_owner(state, reason, state.last_terminated_ts)
            self._push_recent_terminated(state, reason)
            self._push_recent_segment(state, reason, state.last_terminated_ts)
        state.bullet_active        = False
        state.bullet_id            = None
        state.keep_streak          = 0
        state.static_fail_count    = 0
        state.probation_passed     = False
        state.probation_fail_count = 0
        state.probation_until_ts   = -1
        state.probation_ref_pos    = None
        state.probation_start_point_count = 0
        state.probation_start_ts   = -1
        state.birth_assign_kind    = ''
        state.p12_owner_mode = ''
        state.p12_owner_bullet_id = -1
        state.p12_inherited_segment_index = -1
        state.p12_owner_reason = ''
        state.p12_owner_pending = False
        state.p12_owner_pending_mode = ''
        state.p12_owner_pending_since_ts = -1
        state.p12_polluted_tail = False
        # P6: 如果这个 bullet 已经显示过，终止后再保留一小段绘制轨迹。
        # 这只影响画面显示，不让 inactive track 继续参与命中/续接判定。
        if (self._draw_after_terminate_hold_us > 0
                and state.display_bullet_id is not None
                and state.last_terminated_ts >= 0):
            state.hold_display_bullet_id = int(state.display_bullet_id)
            state.hold_display_points = deque(list(state.model_points) if len(state.model_points) >= 2 else list(state.display_points), maxlen=self.draw_history_len)
            state.draw_hold_until_ts = int(state.last_terminated_ts) + int(self._draw_after_terminate_hold_us)
        else:
            state.hold_display_bullet_id = None
            state.hold_display_points.clear()
            state.draw_hold_until_ts = -1
        state.display_bullet_id    = None
        # P2-A: 重置 recent_offset 计数
        state.recent_offset_exceed_count = 0
        state.segment_turn_warmup_until_ts = -1
        if reason in {'bbox_unstable', 'bbox_ema_drift', 'model_outlier'}:
            state.bbox_reuse_min_disp_px = self.same_track_bbox_reuse_min_disp_px
        else:
            state.bbox_reuse_min_disp_px = 0.0
        # P2-C: 不重置 turn_count/cumulative_turn_deg（deactivate 后 track 可能继续存在）
        self._trim_points(state)

    def _try_reuse_same_track_bullet(
        self, state: TrackState, c: dict, ts: int
    ) -> Tuple[Optional[int], str]:
        if state.last_bullet_id is None or state.last_terminated_ts < 0:
            return None, 'none'
        if ts - int(state.last_terminated_ts) > self._same_track_reactivate_window_us:
            return None, 'none'

        # P10-7B safety：model_outlier 后不要走普通 same-track 直接复活，
        # 否则容易把人体/撞击残留粘回旧 bullet。
        # 这类情况只允许 P10-6/P10-7 的 post_outlier_rearm 专用路径使用 last_bullet_id。
        if state.last_terminated_reason in {'model_outlier'}:
            return None, 'none'

        # P6-1: bbox 类终止后，same-track 重用必须先出现新的位移，
        # 避免人体局部框在原地抖动/停住时被立即复活。
        if state.last_terminated_reason in {'bbox_unstable', 'bbox_ema_drift'}:
            if state.last_terminated_pos is None:
                return None, 'none'
            cur_xy = np.array([float(c['cx']), float(c['cy'])], dtype=np.float32)
            prev_xy = np.array(state.last_terminated_pos, dtype=np.float32)
            disp = float(np.linalg.norm(cur_xy - prev_xy))
            min_disp = max(self.same_track_bbox_reuse_min_disp_px, state.bbox_reuse_min_disp_px)
            if disp < min_disp:
                return None, 'none'

        return int(state.last_bullet_id), 'same_track'

    def _try_reuse_recent_terminated_bullet(
        self, state: TrackState, c: dict, ts: int
    ) -> Tuple[Optional[int], str]:
        self._purge_recent_terminated(int(ts))
        cur_xy    = np.array([float(c['cx']), float(c['cy'])], dtype=np.float32)
        state_dir = self._track_direction(state, window_points=min(4, max(2, len(state.points))))
        active_ids = {
            int(s.bullet_id)
            for s in self._tracks.values()
            if s.bullet_active and s.bullet_id is not None
        }
        candidates = []
        for item in self._recent_terminated:
            bullet_id = int(item['bullet_id'])
            if bullet_id in active_ids:
                continue
            dt = int(ts) - int(item['ts'])
            if dt <= 0 or dt > self._ghost_reactivate_window_us:
                continue
            prev_xy    = np.array(item['pos'], dtype=np.float32)
            dist       = float(np.linalg.norm(cur_xy - prev_xy))
            allow_dist = self._ghost_base_distance_px + self._ghost_speed_px_per_us * dt
            if dist > allow_dist:
                continue
            last_w = max(1.0, float(item.get('width',  c['width'])))
            last_h = max(1.0, float(item.get('height', c['height'])))
            cur_w  = max(1.0, float(c['width']))
            cur_h  = max(1.0, float(c['height']))
            sr     = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            if sr > self._maintain_size_ratio:
                continue
            score    = dist + 0.00010 * dt + (sr - 1.0) * 8.0
            ghost_dir = item.get('dir')
            if ghost_dir is not None and state_dir is not None:
                gd_np   = np.array(ghost_dir, dtype=np.float32)
                cos_val = float(np.dot(state_dir, gd_np) / max(np.linalg.norm(gd_np), 1e-6))
                if cos_val >= -0.30:
                    score += (1.0 - abs(max(-1.0, min(1.0, cos_val)))) * (
                        self.association_direction_penalty_px * 0.5
                    )
            candidates.append((score, bullet_id))
        if not candidates:
            return None, 'none'
        candidates.sort(key=lambda x: x[0])
        return int(candidates[0][1]), 'ghost'

    def _try_segment_pool_capture_new_track(
        self, state: TrackState, c: dict, ts: int
    ) -> Tuple[Optional[int], str]:
        """P10-4: recent segment pool 局部续接。

        只对已经有 bullet_id 的近期 segment 生效，窗口比普通 ghost/bounce 稍宽；
        但候选 raw_id 仍需满足短窗弹道特征或非常明确的时空速度关系。
        """
        if not self.ghost_capture_enabled:
            return None, 'none'
        ts_i = int(ts)
        self._purge_recent_segments(ts_i)
        if not self._recent_segments:
            return None, 'none'

        cur_xy = np.array([float(c['cx']), float(c['cy'])], dtype=np.float32)
        cur_w = max(1.0, float(c.get('width', 1.0)))
        cur_h = max(1.0, float(c.get('height', 1.0)))
        active_ids = {
            int(s.bullet_id)
            for s in self._tracks.values()
            if s.bullet_active and s.bullet_id is not None
        }
        handoff_reasons = {'raw_id_sticky_handoff', 'model_outlier', 'impact_candidate', 'missing', 'cleanup_missing'}

        # 候选局部质量：有 4~6 个点时用 ballistic_rearm suffix 检查；
        # 刚出生 1~3 点时，则要求非常明确的位移/速度关系，避免人体慢框误吸。
        br_ok, br_stats, br_info = self._select_ballistic_rearm_suffix_stats(state)
        local_points = len(state.points)
        if br_ok and br_stats is not None:
            local_quality_ok = True
            local_speed_px_ms = float(br_info.get('speed_px_ms', 0.0) or self._estimate_recent_speed_px_per_ms(state))
            local_disp_px = float(br_stats.get('total_disp', 0.0))
        else:
            local_speed_px_ms = float(self._estimate_recent_speed_px_per_ms(state))
            local_disp_px = 0.0
            if local_points >= 2:
                pts = list(state.points)
                local_disp_px = float(math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
            local_quality_ok = bool(
                local_points <= 3
                and local_disp_px >= max(8.0, self.min_total_displacement_px * 0.40)
                and local_speed_px_ms >= max(0.65, self.trigger_min_avg_speed_px_per_ms)
            )
        if not local_quality_ok:
            return None, 'none'

        candidates = []
        for item in self._recent_segments:
            bid = int(item.get('bullet_id', -1))
            # 普通 recent segment 不抢占仍在 active 的 bullet，避免两颗同时飞行被合并；
            # 但 raw_id_sticky_handoff 是旧 raw_id 已经被判定为粘背景，此时允许新 raw_id 继承同一 bullet_id。
            if bid <= 0:
                continue
            if bid in active_ids and str(item.get('reason', '')) not in handoff_reasons:
                continue
            dt = ts_i - int(item.get('ts', ts_i))
            if dt <= 0 or dt > self._segment_bounce_dt_us:
                continue
            prev_xy = np.array(item.get('pos', (0.0, 0.0)), dtype=np.float32)
            dist = float(np.linalg.norm(cur_xy - prev_xy))
            dt_ms = dt / 1000.0
            implied_speed = dist / max(dt_ms, 1e-6)
            prev_speed = float(item.get('recent_speed_px_per_ms', 0.0))
            adaptive_allow = max(
                self._segment_bounce_max_dist_px,
                min(self._segment_bounce_adaptive_max_px, 14.0 + max(prev_speed, local_speed_px_ms) * dt_ms * 0.85),
            )
            if dist > adaptive_allow:
                continue
            if implied_speed < self._segment_bounce_min_speed_px_ms:
                continue

            last_w = max(1.0, float(item.get('width', cur_w)))
            last_h = max(1.0, float(item.get('height', cur_h)))
            sr = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            if sr > self._segment_bounce_max_size_ratio:
                continue

            src = 'segment_bounce'
            dir_penalty = 0.0
            seg_dir = item.get('dir')
            if seg_dir is not None:
                gd = np.array(seg_dir, dtype=np.float32)
                new_vec = cur_xy - prev_xy
                new_norm = float(np.linalg.norm(new_vec))
                gd_norm = float(np.linalg.norm(gd))
                if new_norm > 1e-6 and gd_norm > 1e-6:
                    cos_val = float(np.dot(new_vec / new_norm, gd / gd_norm))
                    # 同向属于短断点直飞续接；明显改变方向属于反弹续接。
                    if cos_val >= 0.70:
                        src = 'segment_forward'
                        dir_penalty = (1.0 - cos_val) * 8.0
                    else:
                        src = 'segment_bounce'
                        # 反弹方向完全随机也不理想，轻微惩罚极端背离/横向不稳定。
                        dir_penalty = max(0.0, abs(cos_val) - 0.95) * 10.0

            score = dist + 0.00012 * dt + (sr - 1.0) * 8.0 + dir_penalty - min(20.0, local_speed_px_ms * 2.0)
            candidates.append((score, bid, src))

        if not candidates:
            return None, 'none'
        candidates.sort(key=lambda x: x[0])
        return int(candidates[0][1]), str(candidates[0][2])

    def _try_ghost_capture_new_track(
        self, c: dict, ts: int
    ) -> Tuple[Optional[int], str]:
        """
        P1/P2-E: Ghost 跨 track 捕获（拆成直飞和反弹两条路径）。

        直飞分支（forward）：
            沿旧方向做前向外推，判断新点离预测点的距离。
            适合子弹连续飞行中出现短暂检测断点的情况。

        反弹分支（bounce）：
            不做前向外推，直接判断新点到旧终止点的原始距离。
            允许方向相反或垂直，门限由 ghost_bounce_max_dist_px 控制。
            适合子弹撞击后反弹、方向反转的情况（b6→r87, b14→b15）。

        返回：(bullet_id 或 None, 来源字符串)
        """
        if not self.ghost_capture_enabled:
            return None, 'none'
        self._purge_recent_terminated(int(ts))
        cur_xy = np.array([float(c['cx']), float(c['cy'])], dtype=np.float32)
        active_ids = {
            int(s.bullet_id)
            for s in self._tracks.values()
            if s.bullet_active and s.bullet_id is not None
        }
        cur_w = max(1.0, float(c['width']))
        cur_h = max(1.0, float(c['height']))

        forward_candidates = []
        bounce_candidates  = []

        for item in self._recent_terminated:
            bullet_id = int(item['bullet_id'])
            if bullet_id in active_ids:
                continue
            dt = int(ts) - int(item['ts'])
            if dt <= 0 or dt > 100_000:    # P3-1: 300ms → 100ms
                continue
            prev_xy   = np.array(item['pos'], dtype=np.float32)
            ghost_dir = item.get('dir')

            last_w = max(1.0, float(item.get('width',  cur_w)))
            last_h = max(1.0, float(item.get('height', cur_h)))
            sr     = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            if sr > self._maintain_size_ratio:
                continue

            # --- 直飞分支 ---
            if ghost_dir is not None:
                gd = np.array(ghost_dir, dtype=np.float32)
                speed_px_per_us = 0.0012
                pred_xy = prev_xy + gd * speed_px_per_us * dt
                dist_to_pred = float(np.linalg.norm(cur_xy - pred_xy))
                allow_fwd = 40.0 + 0.00015 * dt
                if dist_to_pred <= allow_fwd:
                    score = dist_to_pred + 0.0001 * dt
                    forward_candidates.append((score, bullet_id, 'ghost_capture'))

            # --- 反弹分支（P6）：终止点原始距离 + 自适应距离 + px/ms 速度门槛 ---
            dist_to_term = float(np.linalg.norm(cur_xy - prev_xy))
            dt_ms = dt / 1000.0
            implied_speed_px_per_ms = dist_to_term / max(dt_ms, 1e-6)
            prev_speed_px_per_ms = float(item.get('recent_speed_px_per_ms', 0.0))
            # 终止前速度越快，允许的反弹续接距离越大；上限做裁剪，避免过度放宽。
            adaptive_allow_bounce = max(
                self.ghost_bounce_max_dist_px,
                min(140.0, 12.0 + max(0.0, prev_speed_px_per_ms) * dt_ms * 0.90)
            )
            if (dist_to_term <= adaptive_allow_bounce
                    and implied_speed_px_per_ms >= self.ghost_bounce_min_speed_px_per_ms):
                # 方向检查：方向相同（直飞）→ 跳过反弹分支，让直飞分支处理
                skip_bounce = False
                if ghost_dir is not None:
                    gd = np.array(ghost_dir, dtype=np.float32)
                    new_vec = cur_xy - prev_xy
                    new_norm = float(np.linalg.norm(new_vec))
                    if new_norm > 1e-6:
                        new_dir_est = new_vec / new_norm
                        cos_val = float(np.dot(new_dir_est, gd / max(np.linalg.norm(gd), 1e-6)))
                        if cos_val >= 0.7:
                            skip_bounce = True
                if not skip_bounce:
                    score = dist_to_term + 0.0002 * dt + 20.0
                    bounce_candidates.append((score, bullet_id, 'ghost_capture_bounce'))

        # 合并：直飞优先，反弹补充；对同一 bullet_id 只取最优分支
        all_candidates: Dict[int, Tuple[float, str]] = {}
        for score, bid, src in forward_candidates:
            if bid not in all_candidates or score < all_candidates[bid][0]:
                all_candidates[bid] = (score, src)
        for score, bid, src in bounce_candidates:
            if bid not in all_candidates or score < all_candidates[bid][0]:
                all_candidates[bid] = (score, src)

        if not all_candidates:
            return None, 'none'
        best_bid = min(all_candidates, key=lambda b: all_candidates[b][0])
        return int(best_bid), all_candidates[best_bid][1]

    # -----------------------------------------------------------------------
    # P2-F: 判断某个 track 的起始点是否在疑似续接窗口内
    # -----------------------------------------------------------------------
    def _is_suspected_continuation(self, state: TrackState, ts: int) -> bool:
        """
        检查该 track 的首个观测是否落在某个已终止 bullet 的时空续接窗口内。
        同时检查直飞和反弹两套距离标准，以及尺寸约束。
        返回 True 表示应豁免 late_trigger_veto。
        """
        if state.last_cluster is None:
            return False
        self._purge_recent_terminated(int(ts))
        first_pos = np.array([float(state.last_cluster['cx']), float(state.last_cluster['cy'])],
                             dtype=np.float32)
        cur_w = max(1.0, float(state.last_cluster.get('width', 1.0)))
        cur_h = max(1.0, float(state.last_cluster.get('height', 1.0)))

        for item in self._recent_terminated:
            dt = int(ts) - int(item['ts'])
            if dt <= 0 or dt > self._late_trigger_exempt_time_us:
                continue
            last_w = max(1.0, float(item.get('width',  cur_w)))
            last_h = max(1.0, float(item.get('height', cur_h)))
            sr     = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
            if sr > self._maintain_size_ratio:
                continue
            prev_xy   = np.array(item['pos'], dtype=np.float32)
            ghost_dir = item.get('dir')
            # 直飞分支：前向外推距离
            if ghost_dir is not None:
                gd = np.array(ghost_dir, dtype=np.float32)
                pred_xy = prev_xy + gd * 0.0012 * dt
                dist_fwd = float(np.linalg.norm(first_pos - pred_xy))
                if dist_fwd <= self._late_trigger_exempt_fwd_px:
                    return True
            # 反弹分支：终止点原始距离
            dist_raw = float(np.linalg.norm(first_pos - prev_xy))
            if dist_raw <= self._late_trigger_exempt_bounce_px:
                return True
        return False

    def _slice_stats(
        self, pts: List[Tuple[float, float, float, float, int]]
    ) -> Optional[dict]:
        n = len(pts)
        if n == 0:
            return None
        xy = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        step_vecs  = xy[1:] - xy[:-1]
        if len(step_vecs) == 0:
            return {
                'n': n, 'xy': xy,
                'total_disp': 0.0, 'max_offset': 0.0, 'mean_offset': 0.0,
                'direction_ok_ratio': 0.0, 'same_sign_ratio': 0.0,
                'total_step_count': 0, 'valid_step_count': 0, 'valid_step_ratio': 0.0,
                'direction': None, 'path_length': 0.0, 'path_over_net': 999.0,
                'sign_flip_ratio': 0.0,   # P3-5
            }

        step_norms  = np.linalg.norm(step_vecs, axis=1)
        valid_mask  = step_norms >= self.min_step_px
        valid_steps = step_vecs[valid_mask] if np.any(valid_mask) else np.empty((0, 2), dtype=np.float32)
        valid_norms = np.linalg.norm(valid_steps, axis=1) if len(valid_steps) else np.empty((0,))

        total_disp   = float(np.linalg.norm(xy[-1] - xy[0]))
        path_length  = float(np.sum(step_norms))
        path_over_net = float(path_length / max(total_disp, 1e-6))

        mean_xy  = xy.mean(axis=0)
        centered = xy - mean_xy
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction      = vh[0]
            direction_norm = np.linalg.norm(direction)
        except np.linalg.LinAlgError:
            direction      = np.array([1.0, 0.0], dtype=np.float32)
            direction_norm = 1.0
        if direction_norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / direction_norm

        rel     = xy - mean_xy
        proj    = rel @ direction
        closest = np.outer(proj, direction) + mean_xy
        offsets = np.linalg.norm(xy - closest, axis=1)
        max_offset  = float(np.max(offsets))  if len(offsets) else 0.0
        mean_offset = float(np.mean(offsets)) if len(offsets) else 0.0

        total_step_count = int(len(step_vecs))
        valid_step_count = int(len(valid_steps))
        valid_step_ratio = float(valid_step_count / max(total_step_count, 1))

        if len(valid_steps):
            cos_thresh        = math.cos(math.radians(self.max_angle_deg))
            cos_vals          = np.abs((valid_steps @ direction) / np.maximum(valid_norms, 1e-6))
            direction_ok_ratio = float(np.mean(cos_vals >= cos_thresh))
            proj_vals          = valid_steps @ direction
            positive_ratio     = float(np.mean(proj_vals >=  self.min_step_px))
            negative_ratio     = float(np.mean(proj_vals <= -self.min_step_px))
            same_sign_ratio    = max(positive_ratio, negative_ratio)
        else:
            direction_ok_ratio = 0.0
            same_sign_ratio    = 0.0

        # P3-5: 相邻步方向反转比率
        if len(step_vecs) >= 2:
            dots = np.array([
                float(np.dot(step_vecs[i], step_vecs[i + 1]))
                for i in range(len(step_vecs) - 1)
            ], dtype=np.float32)
            sign_flip_ratio = float(np.mean(dots < 0))
        else:
            sign_flip_ratio = 0.0

        return {
            'n': n, 'xy': xy,
            'total_disp': total_disp, 'max_offset': max_offset, 'mean_offset': mean_offset,
            'direction_ok_ratio': direction_ok_ratio, 'same_sign_ratio': same_sign_ratio,
            'total_step_count': total_step_count, 'valid_step_count': valid_step_count,
            'valid_step_ratio': valid_step_ratio,
            'direction': direction, 'path_length': path_length, 'path_over_net': path_over_net,
            'sign_flip_ratio': sign_flip_ratio,   # P3-5
        }

    def _compute_stats(
        self, state: TrackState
    ) -> Tuple[Optional[dict], Optional[dict]]:
        pts = list(state.points)
        if not pts:
            return None, None
        return (
            self._slice_stats(pts),
            self._slice_stats(pts[-self._recent_window_points:]),
        )

    def _estimate_speed_from_points(self, pts: List[Tuple[float, float, float, float, int]]) -> float:
        if pts is None or len(pts) < 2:
            return 0.0
        t0 = int(pts[0][4])
        t1 = int(pts[-1][4])
        dt_ms = (t1 - t0) / 1000.0
        if dt_ms <= 0:
            return 0.0
        p0 = np.array([pts[0][0], pts[0][1]], dtype=np.float32)
        p1 = np.array([pts[-1][0], pts[-1][1]], dtype=np.float32)
        return float(np.linalg.norm(p1 - p0) / dt_ms)

    def _estimate_recent_speed_px_per_ms(
        self, state: TrackState, keep_points: Optional[int] = None
    ) -> float:
        pts = list(state.points)
        if len(pts) < 2:
            return 0.0
        k = self._recent_window_points if keep_points is None else max(2, int(keep_points))
        return self._estimate_speed_from_points(pts[-k:])

    def _promote_display_id(self, state: TrackState) -> None:
        if state.bullet_id is None:
            return
        state.display_bullet_id = int(state.bullet_id)
        self._displayed_bullet_ids.add(int(state.bullet_id))

    # -------------------------------------------------------------------
    # P10-7A: pre-probation early display promotion
    # -------------------------------------------------------------------
    def _try_promote_display_early(
        self, state: TrackState, recent_stats: Optional[dict], ts: int
    ) -> bool:
        """
        P10-7A：短存活高质量子弹段提前进入显示层。

        背景：P10-5 已经能把确认前的可信候选点回填到 model/display 历史，
        但 display_bullet_id 仍然通常要等 probation_passed 后才提升。
        raw183 这类“trigger 后很快 impact/model_outlier”的段，内部已经有 bullet_id
        和 effect/backfill 行，却可能因为没完整跑完 probation 而没有黄色显示。

        这里不降低 trigger/probation 判定门槛，也不提前发高置信命中；
        只在 bullet_id 已经分配、最近短窗质量足够高时，把 display_bullet_id 提前设置。
        """
        if state.bullet_id is None or (not state.bullet_active):
            return False
        if state.display_bullet_id is not None:
            return False
        if state.probation_passed:
            return False
        if recent_stats is None:
            return False

        # P12: 在已确认子弹的 impact/occlusion 保护区内，
        # ballistic_rearm / bounce 类候选不能独立提前显示；
        # 只有 owner_forward / owner_bounce 这种继承旧 bullet_id 的段允许显示。
        risky_birth = str(state.birth_assign_kind) in {
            'ballistic_rearm', 'post_outlier_rearm',
            'ghost_capture_bounce', 'segment_bounce', 'ghost_capture', 'segment_forward'
        }
        if risky_birth and str(getattr(state, 'p12_owner_mode', '')) not in {'owner_forward', 'owner_bounce'}:
            if state.last_cluster is not None and self._is_near_occlusion_owner(state.last_cluster, int(ts)):
                return False

        # P10-5 回填点可以证明“候选弹道已持续存在”；不能只看 bullet_assign_ts，
        # 否则 confirmed_backfill 场景会被低估存活时间。
        evidence_start_ts = int(state.bullet_assign_ts) if state.bullet_assign_ts >= 0 else int(ts)
        backfill_n = 0
        try:
            bf = list(getattr(state, 'confirmed_backfill_points', []))
            backfill_n = len(bf)
            if bf:
                evidence_start_ts = min(evidence_start_ts, min(int(p[2]) for p in bf))
        except Exception:
            backfill_n = 0

        elapsed_us = int(ts) - int(evidence_start_ts)
        if elapsed_us < 12_000:
            return False

        # 有 confirmed_backfill 支撑时，keep_streak 可以略低；否则仍要求至少 3 帧。
        min_keep = 2 if backfill_n >= 3 else 3
        if int(state.keep_streak) < min_keep:
            return False

        total_disp = float(recent_stats.get('total_disp', 0.0))
        max_offset = float(recent_stats.get('max_offset', 999.0))
        dir_ratio = float(recent_stats.get('direction_ok_ratio', 0.0))
        same_sign = float(recent_stats.get('same_sign_ratio', 0.0))
        flip_ratio = float(recent_stats.get('sign_flip_ratio', 1.0))
        path_over_net = float(recent_stats.get('path_over_net', 999.0))
        valid_steps = int(recent_stats.get('valid_step_count', 0))
        speed = float(self._estimate_recent_speed_px_per_ms(state))

        # 对“显示提前”比 trigger 略宽，但仍要求像高速直线；
        # 尤其控制 offset/flip，避免人体残留被提前显示成黄线。
        ok = bool(
            total_disp >= max(10.0, self.min_total_displacement_px * 0.50)
            and max_offset <= self.max_line_offset_px * 1.50
            and dir_ratio >= 0.68
            and same_sign >= 0.68
            and flip_ratio <= 0.38
            and path_over_net <= 2.35
            and valid_steps >= 1
            and speed >= max(0.35, self.trigger_min_avg_speed_px_per_ms * 0.55)
        )
        if not ok:
            return False

        self._promote_display_id(state)
        return True

    def _post_assign_stats(self, state: TrackState) -> Optional[dict]:
        if state.bullet_assign_ts < 0:
            return None
        # P8: 试用期新增运动优先看 model_points；obs 点可能被人体框拉偏或停留。
        if state.model_ready and len(state.model_points) >= 2:
            mpts = [p for p in state.model_points if int(p[2]) >= int(state.bullet_assign_ts)]
            if len(mpts) >= 2:
                pseudo = [(float(x), float(y), 1.0, 1.0, int(t)) for x, y, t in mpts]
                return self._slice_stats(pseudo)
        pts = [p for p in state.points if int(p[4]) >= int(state.bullet_assign_ts)]
        if len(pts) < 2:
            return None
        return self._slice_stats(pts)

    def _required_trigger_streak(self, recent_stats: Optional[dict]) -> int:
        required = int(self.bullet_min_output_streak)
        if recent_stats is None:
            return required
        very_clean = (
            recent_stats.get('direction_ok_ratio', 0.0) >= 0.95
            and recent_stats.get('same_sign_ratio', 0.0) >= 0.95
            and recent_stats.get('max_offset', 999.0) <= min(1.5, self.max_line_offset_px * 0.45)
            and recent_stats.get('total_disp', 0.0) >= max(35.0, self.min_total_displacement_px * 1.2)
            and recent_stats.get('path_over_net', 999.0) <= 1.15
            and recent_stats.get('sign_flip_ratio', 1.0) <= 0.05
        )
        if very_clean:
            required = min(required, 3)
        return required


    # -----------------------------------------------------------------------
    # P10-6: post-outlier rearm / raw 内部分段重启
    # -----------------------------------------------------------------------
    def _select_post_outlier_rearm_suffix_stats(self, state: TrackState, ts: int) -> Tuple[bool, Optional[dict], dict]:
        """
        P10-6: 一个 track 刚因 model_outlier / probation 撤销后，如果同一个
        raw/stable track 后续又出现高质量 4~6 帧弹道短窗，则允许重新开一段。

        注意：这不是全局放宽 trigger。必须满足：
        - 之前已经有 last_bullet_id；
        - 终止原因属于 impact/probation 类；
        - 距离上次终止时间在短窗口内；
        - 当前后缀窗口通过 ballistic_rearm 的高速直线检查。
        """
        if not self._post_outlier_rearm_enabled:
            return False, None, {'reason': 'disabled'}
        if state.last_bullet_id is None or state.last_terminated_ts < 0:
            return False, None, {'reason': 'no_last_bullet'}
        reason = str(state.last_terminated_reason or '')
        if reason not in self._post_outlier_rearm_reasons:
            return False, None, {'reason': 'reason_not_allowed', 'last_reason': reason}
        dt = int(ts) - int(state.last_terminated_ts)
        if dt < self._post_outlier_rearm_min_dt_us:
            return False, None, {'reason': 'too_early', 'dt_us': int(dt)}
        if dt > self._post_outlier_rearm_window_us:
            return False, None, {'reason': 'too_late', 'dt_us': int(dt)}

        ok, stats, info = self._select_ballistic_rearm_suffix_stats(state)
        info = dict(info or {})
        info['post_dt_us'] = int(dt)
        info['last_reason'] = reason
        if not ok or stats is None:
            info.setdefault('reason', 'ballistic_suffix_fail')
            return False, None, info
        info['reason'] = 'ok'
        return True, stats, info

    def _reset_points_to_recent_suffix_for_rearm(self, state: TrackState, window: int) -> None:
        """P10-6: post-outlier rearm 成功时，将判断窗口重置为最近可信短窗。

        这样 raw70 / raw203 这类长 raw_id 不会继续被老的静止点、翻转点、
        人体残留点污染。这里只保留真实检测到的最近短窗，不做未来外推。
        """
        if not self._post_outlier_rearm_reset_points:
            return
        pts = list(state.points)
        if not pts:
            return
        k = max(self._ballistic_rearm_window_min, min(int(window or 0), self._ballistic_rearm_window_max))
        if len(pts) < k:
            k = len(pts)
        suffix = pts[-k:]
        if len(suffix) < 2:
            return

        state.points = deque(suffix, maxlen=self.history_len)
        state.pre_confirm_points = deque(
            [(float(x), float(y), int(t)) for x, y, _w, _h, t in suffix],
            maxlen=self.pre_confirm_history_len,
        )
        state.obs_points = deque(
            [(float(x), float(y), int(t)) for x, y, _w, _h, t in suffix],
            maxlen=48,
        )
        state.display_points = deque(maxlen=self.draw_history_len)
        state.model_points.clear()
        state.model_ready = False
        state.model_last_ts = -1
        state.model_update_mode = 'post_outlier_rearm_reset'
        state.recent_widths = deque([float(w) for _x, _y, w, _h, _t in suffix], maxlen=8)
        state.recent_heights = deque([float(h) for _x, _y, _w, h, _t in suffix], maxlen=8)


    def _select_ballistic_rearm_suffix_stats(self, state: TrackState) -> Tuple[bool, Optional[dict], dict]:
        """
        P10-1: late_trigger_veto 的局部弹道逃生通道。

        与离线 ballistic_rearm_csv_backtest_v3_nopandas.py 的 --suffix-only 思路一致：
        不在 raw_id 生命周期里枚举任意最优窗口，只检查当前 state.points 的最后 4~6 个后缀窗口。
        这样能让“老 raw_id 最近突然出现高速直线段”的真实子弹重新进入 trigger/probation，
        同时不全局放宽普通 trigger 条件。
        """
        if not self._ballistic_rearm_enabled:
            return False, None, {'reason': 'disabled'}

        pts_all = list(state.points)
        if len(pts_all) < self._ballistic_rearm_window_min:
            return False, None, {'reason': 'not_enough_points'}

        best_stats: Optional[dict] = None
        best_info: dict = {'reason': 'no_window'}
        best_score = -1e9

        # 只检查后缀窗口：[-4:]、[-5:]、[-6:]。
        for k in range(self._ballistic_rearm_window_min, self._ballistic_rearm_window_max + 1):
            if len(pts_all) < k:
                continue
            pts = pts_all[-k:]
            stats = self._slice_stats(pts)
            if stats is None:
                continue

            speed = self._estimate_speed_from_points(pts)
            widths = [max(1.0, float(p[2])) for p in pts]
            heights = [max(1.0, float(p[3])) for p in pts]
            longs = [max(w, h) for w, h in zip(widths, heights)]
            areas = [w * h for w, h in zip(widths, heights)]
            try:
                bbox_area_median = float(np.median(np.asarray(areas, dtype=np.float32)))
                bbox_long_median = float(np.median(np.asarray(longs, dtype=np.float32)))
            except Exception:
                bbox_area_median = float(sum(areas) / max(1, len(areas)))
                bbox_long_median = float(sum(longs) / max(1, len(longs)))
            bbox_area_last = float(areas[-1])
            bbox_long_last = float(longs[-1])

            fail = []
            disp_th = max(8.0, self.min_total_displacement_px * self._ballistic_rearm_disp_ratio)
            offset_th = self.max_line_offset_px * self._ballistic_rearm_offset_ratio

            if float(stats.get('total_disp', 0.0)) < disp_th:
                fail.append('disp')
            if speed < self.trigger_min_avg_speed_px_per_ms:
                fail.append('speed')
            if int(stats.get('valid_step_count', 0)) < self._ballistic_rearm_min_valid_steps:
                fail.append('valid_steps')
            if float(stats.get('path_over_net', 999.0)) > self._ballistic_rearm_max_path_over_net:
                fail.append('path_over_net')
            if float(stats.get('max_offset', 999.0)) > offset_th:
                fail.append('offset')
            if float(stats.get('direction_ok_ratio', 0.0)) < self._ballistic_rearm_min_direction_ok_ratio:
                fail.append('direction')
            if float(stats.get('same_sign_ratio', 0.0)) < self._ballistic_rearm_min_same_sign_ratio:
                fail.append('same_sign')
            if float(stats.get('sign_flip_ratio', 1.0)) > self._ballistic_rearm_max_sign_flip_ratio:
                fail.append('flip')
            # median + last 双重保护：median 排除整体人体框，last 排除当前帧已粘到人体的大框。
            if bbox_area_median > self._ballistic_rearm_max_bbox_area:
                fail.append('bbox_area_median')
            if bbox_long_median > self._ballistic_rearm_max_bbox_long_side:
                fail.append('bbox_long_median')
            if bbox_area_last > self._ballistic_rearm_max_bbox_area:
                fail.append('bbox_area_last')
            if bbox_long_last > self._ballistic_rearm_max_bbox_long_side:
                fail.append('bbox_long_last')

            score = 0.0
            score += min(2.0, float(stats.get('total_disp', 0.0)) / max(1e-6, disp_th))
            score += min(2.0, speed / max(1e-6, self.trigger_min_avg_speed_px_per_ms))
            score += min(1.5, float(stats.get('valid_step_count', 0)) / max(1, self._ballistic_rearm_min_valid_steps))
            score += min(1.5, self._ballistic_rearm_max_path_over_net / max(1e-6, float(stats.get('path_over_net', 999.0))))
            score += min(1.0, offset_th / max(1e-6, float(stats.get('max_offset', 999.0))))
            score += float(stats.get('direction_ok_ratio', 0.0))
            score += float(stats.get('same_sign_ratio', 0.0))
            score += max(0.0, 1.0 - float(stats.get('sign_flip_ratio', 1.0)))

            info = {
                'window': int(k),
                'speed_px_ms': float(speed),
                'disp_thresh': float(disp_th),
                'offset_thresh': float(offset_th),
                'bbox_area_median': float(bbox_area_median),
                'bbox_long_median': float(bbox_long_median),
                'bbox_area_last': float(bbox_area_last),
                'bbox_long_last': float(bbox_long_last),
                'fail': '+'.join(fail) if fail else 'ok',
                'score': float(score),
            }
            if not fail and score > best_score:
                best_score = score
                best_stats = stats
                best_info = info

        if best_stats is None:
            return False, None, best_info
        return True, best_stats, best_info

    def _evaluate_trigger(
        self, state: TrackState, full_stats: Optional[dict],
        recent_stats: Optional[dict], assign_kind: str, ts: int
    ) -> dict:
        _empty = {
            'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
            'confirmed_now': False, 'phase': 'none', 'reject_reason': 'none',
            'line_ok': False, 'direction_ok': False, 'motion_ok': False,
            'recent_geom_ok': False, 'recent_motion_ok': False,
            'trigger_disp_thresh': float(self.min_total_displacement_px), 'compact_ok': False,
            'rearm_mode': False, 'ballistic_rearm': False,
            'ballistic_rearm_window': 0, 'ballistic_rearm_speed_px_ms': 0.0,
            'post_outlier_rearm': False, 'post_outlier_rearm_window': 0,
            'post_outlier_rearm_dt_us': 0,
            'required_trigger_streak_override': 0,
        }
        if full_stats is None or recent_stats is None:
            return _empty

        pts = list(state.points)
        if len(pts) <= self.bootstrap_frames:
            return {**_empty, 'phase': 'bootstrap', 'reject_reason': 'bootstrap'}

        track_age_us = int(ts) - int(state.first_seen_ts)

        # P2-F: 疑似续接目标豁免 late_trigger
        suspected_continuation = self._is_suspected_continuation(state, int(ts))

        late_trigger_veto = (
            not state.confirmed_once
            and state.last_bullet_id is None
            and track_age_us > self._late_trigger_max_age_us
            and not suspected_continuation
        )

        # P5-2: recent-clean rearm + P10-1: recent ballistic rearm
        rearm_mode = False
        ballistic_rearm = False
        post_outlier_rearm = False
        ballistic_rearm_info = {'window': 0, 'speed_px_ms': 0.0, 'fail': ''}

        # P10-6: post-outlier rearm 优先级高于普通 late_trigger。
        # 只针对已经有 last_bullet_id 且刚因 impact/probation 类原因终止的同一 track。
        por_ok, por_stats, por_info = self._select_post_outlier_rearm_suffix_stats(state, int(ts))
        if por_ok and por_stats is not None:
            post_outlier_rearm = True
            ballistic_rearm = True
            rearm_mode = True
            recent_stats = por_stats
            ballistic_rearm_info = dict(por_info or {})
            late_trigger_veto = False

        if (not post_outlier_rearm) and late_trigger_veto and recent_stats is not None:
            rearm_mode = (
                recent_stats['direction_ok_ratio'] >= 0.90
                and recent_stats['same_sign_ratio'] >= 0.90
                and recent_stats['path_over_net'] <= 1.30
                and recent_stats['max_offset'] <= self.max_line_offset_px
                and recent_stats['total_disp'] >= max(12.0, self.min_total_displacement_px * 0.60)
                and recent_stats.get('sign_flip_ratio', 1.0) <= 0.10
            )
            if rearm_mode:
                late_trigger_veto = False
            else:
                # P10-1: 老 raw_id 最近 4~6 帧突然出现高速直线段时，允许绕过 late_trigger_veto。
                # 注意：这里不直接分配 bullet_id，只让它进入 trigger，并通过 keep_streak/probation 再确认。
                br_ok, br_stats, br_info = self._select_ballistic_rearm_suffix_stats(state)
                ballistic_rearm_info = br_info
                if br_ok and br_stats is not None:
                    ballistic_rearm = True
                    rearm_mode = True
                    recent_stats = br_stats
                    late_trigger_veto = False

        if ballistic_rearm:
            line_ok = recent_stats['max_offset'] <= self.max_line_offset_px * self._ballistic_rearm_offset_ratio
            direction_ok = (
                recent_stats['direction_ok_ratio'] >= self._ballistic_rearm_min_direction_ok_ratio
                and recent_stats['same_sign_ratio'] >= self._ballistic_rearm_min_same_sign_ratio
            )
        else:
            line_ok = recent_stats['max_offset'] <= self._trigger_max_recent_offset_px
            direction_ok = (
                recent_stats['direction_ok_ratio'] >= self._trigger_min_dir_ratio
                and recent_stats['same_sign_ratio'] >= self._trigger_min_same_sign
            )

        continuation_geom_ok = (
            full_stats['max_offset'] <= self.max_line_offset_px * 1.2
        )
        effective_suspected = suspected_continuation and continuation_geom_ok

        # rearm / suspected 时，对几何和翻转优先使用 recent 口径
        effective_offset_stats = recent_stats if (rearm_mode or effective_suspected) else full_stats
        effective_flip_stats   = recent_stats if (rearm_mode or effective_suspected) else full_stats

        if ballistic_rearm:
            full_offset_ok = (
                effective_offset_stats['max_offset'] <= self.max_line_offset_px * self._ballistic_rearm_offset_ratio
            )
            flip_ok = (
                effective_flip_stats.get('sign_flip_ratio', 0.0) <= self._ballistic_rearm_max_sign_flip_ratio
            )
        else:
            full_offset_ok = (
                effective_offset_stats['max_offset'] <= self.max_line_offset_px * self.trigger_max_full_offset_ratio
            )
            flip_ok = (
                effective_flip_stats.get('sign_flip_ratio', 0.0) <= self.trigger_max_sign_flip_ratio
            )

        # P2-F: 对疑似续接目标降低 motion 要求，但不改变 direction_ok
        if effective_suspected:
            min_valid_steps = max(1, self._trigger_min_recent_valid_steps - 1)
            min_valid_ratio = max(0.40, self._trigger_min_recent_valid_ratio - 0.15)
        else:
            min_valid_steps = self._trigger_min_recent_valid_steps
            min_valid_ratio = self._trigger_min_recent_valid_ratio

        motion_ok = (
            recent_stats['valid_step_count'] >= min_valid_steps
            and recent_stats['valid_step_ratio'] >= min_valid_ratio
        )
        compact_ok = recent_stats['path_over_net'] <= self._trigger_max_path_over_net

        track_age_ms = (int(ts) - int(state.first_seen_ts)) / 1000.0 if state.first_seen_ts >= 0 else 0.0
        disp_per_step = float(recent_stats['total_disp']) / max(1, int(recent_stats['valid_step_count']))

        # P7-1: 仅对 raw_id 新生 trigger 做条件性收紧。
        is_raw_birth = (assign_kind == 'raw_id') and (not effective_suspected) and (not rearm_mode)
        sparse_burst_veto = bool(
            is_raw_birth
            and recent_stats['valid_step_count'] <= self.trigger_sparse_burst_max_valid_steps
            and recent_stats['total_disp'] >= self.trigger_sparse_burst_min_total_disp_px
            and disp_per_step >= self.trigger_sparse_burst_min_disp_per_step_px
            and recent_stats['path_over_net'] <= self.trigger_sparse_burst_max_path_over_net
        )
        raw_birth_steps_ok = bool((not is_raw_birth) or recent_stats['valid_step_count'] >= self.trigger_raw_min_valid_steps)
        old_raw_steps_ok = bool(
            (not is_raw_birth)
            or track_age_ms < self.trigger_old_raw_age_ms
            or recent_stats['valid_step_count'] >= self.trigger_old_raw_min_valid_steps
        )

        # P5-1: recent speed 优先，避免老历史把后段真子弹平均慢
        full_avg_speed = 0.0
        recent_avg_speed = self._estimate_recent_speed_px_per_ms(state)
        speed_ok = True
        if self.trigger_min_avg_speed_px_per_ms > 0:
            if track_age_ms > 0:
                full_avg_speed = full_stats['total_disp'] / track_age_ms
            if ballistic_rearm:
                recent_avg_speed = float(ballistic_rearm_info.get('speed_px_ms', recent_avg_speed) or recent_avg_speed)
                speed_ok = recent_avg_speed >= self.trigger_min_avg_speed_px_per_ms
            elif effective_suspected or rearm_mode:
                speed_ok = recent_avg_speed >= self.trigger_min_avg_speed_px_per_ms
            else:
                speed_ok = max(full_avg_speed, recent_avg_speed) >= self.trigger_min_avg_speed_px_per_ms

        trigger_disp_thresh = float(self.min_total_displacement_px)
        if recent_stats['direction_ok_ratio'] >= 0.82 and recent_stats['max_offset'] <= self.max_line_offset_px:
            trigger_disp_thresh = min(trigger_disp_thresh, max(10.0, self.min_total_displacement_px * 0.72))
        if assign_kind == 'assoc':
            trigger_disp_thresh = min(trigger_disp_thresh, max(10.0, self.min_total_displacement_px * 0.80))
        if effective_suspected:
            trigger_disp_thresh = min(trigger_disp_thresh, max(8.0, self.min_total_displacement_px * 0.60))
        if rearm_mode:
            trigger_disp_thresh = min(trigger_disp_thresh, max(10.0, self.min_total_displacement_px * 0.65))

        trigger_ok = (
            not late_trigger_veto
            and not sparse_burst_veto
            and raw_birth_steps_ok
            and old_raw_steps_ok
            and recent_stats['total_disp'] >= trigger_disp_thresh
            and line_ok and direction_ok and motion_ok and compact_ok
            and full_offset_ok
            and flip_ok
            and speed_ok
        )

        if trigger_ok:
            reject_reason = 'ok'
            if post_outlier_rearm:
                phase = 'trigger_post_outlier_rearm'
            else:
                phase = 'trigger_ballistic_rearm' if ballistic_rearm else ('trigger_rearm' if rearm_mode else 'trigger')
            keep_now      = True
        else:
            reasons = []
            if late_trigger_veto:                                  reasons.append('late_trigger')
            if sparse_burst_veto:                                  reasons.append('sparse_burst')
            if not raw_birth_steps_ok:                             reasons.append('raw_steps')
            if not old_raw_steps_ok:                               reasons.append('old_raw_steps')
            if recent_stats['total_disp'] < trigger_disp_thresh:   reasons.append('disp')
            if not line_ok:                                        reasons.append('line')
            if not direction_ok:                                   reasons.append('dir')
            if not motion_ok:                                      reasons.append('motion')
            if not compact_ok:                                     reasons.append('compact')
            if not full_offset_ok:                                 reasons.append('full_offset')
            if not flip_ok:                                        reasons.append('flip')
            if not speed_ok:                                       reasons.append('slow')
            reject_reason = '+'.join(reasons) if reasons else 'trigger_fail'
            phase         = 'trigger_fail'
            keep_now      = False

        return {
            'keep_now': keep_now, 'trigger_ok': trigger_ok, 'maintain_ok': False,
            'confirmed_now': trigger_ok, 'phase': phase, 'reject_reason': reject_reason,
            'line_ok': line_ok, 'direction_ok': direction_ok, 'motion_ok': motion_ok,
            'recent_geom_ok': line_ok, 'recent_motion_ok': motion_ok,
            'trigger_disp_thresh': float(trigger_disp_thresh), 'compact_ok': compact_ok,
            'rearm_mode': bool(rearm_mode),
            'ballistic_rearm': bool(ballistic_rearm),
            'ballistic_rearm_window': int(ballistic_rearm_info.get('window', 0) or 0),
            'ballistic_rearm_speed_px_ms': float(ballistic_rearm_info.get('speed_px_ms', 0.0) or 0.0),
            'post_outlier_rearm': bool(post_outlier_rearm),
            'post_outlier_rearm_window': int(ballistic_rearm_info.get('window', 0) or 0) if post_outlier_rearm else 0,
            'post_outlier_rearm_dt_us': int(ballistic_rearm_info.get('post_dt_us', 0) or 0) if post_outlier_rearm else 0,
            'required_trigger_streak_override': int(
                self._post_outlier_rearm_required_streak if post_outlier_rearm
                else (self._ballistic_rearm_required_streak if ballistic_rearm else 0)
            ),
        }

    def _size_ok_for_maintain(self, state: TrackState, c: dict) -> bool:
        last = state.last_cluster
        if last is None:
            return True
        last_w = max(1.0, float(last['width']))
        last_h = max(1.0, float(last['height']))
        cur_w  = max(1.0, float(c['width']))
        cur_h  = max(1.0, float(c['height']))
        sr     = max(cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h)
        return sr <= self._maintain_size_ratio

    def _update_segment_state(self, state: TrackState, recent_stats: Optional[dict], ts: int) -> str:
        """
        P2-C/D: 检测方向转折。
        若发生 turn：
          1. 累加 turn_count 和 cumulative_turn_deg
          2. 把 state.points 截断到仅保留当前段的点（keep_tail_points 个），
             让后续 full_stats 只反映当前段的直线质量。

        P5-3: reuse / ghost 刚续接后的短暂 warmup 窗口内，不累计 turn，
        只更新 segment_dir，避免真实反弹刚续上就被 overturn 提前 kill。
        """
        if recent_stats is None or recent_stats.get('direction') is None:
            return 'maintain'
        cur_dir = recent_stats['direction']

        if (
            state.segment_turn_warmup_until_ts >= 0
            and int(ts) <= int(state.segment_turn_warmup_until_ts)
        ):
            state.segment_dir = (float(cur_dir[0]), float(cur_dir[1]))
            return 'maintain'

        if state.segment_dir is None:
            state.segment_dir = (float(cur_dir[0]), float(cur_dir[1]))
            return 'maintain'
        prev_dir = np.array(state.segment_dir, dtype=np.float32)
        cos_val  = float(np.dot(cur_dir, prev_dir) / max(np.linalg.norm(prev_dir), 1e-6))
        if recent_stats['total_disp'] >= max(6.0, self.min_step_px * 1.6) and cos_val <= self._segment_turn_cos:
            turn_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))
            state.turn_count          += 1
            state.cumulative_turn_deg += turn_deg
            state.segment_index       += 1
            state.segment_dir          = (float(cur_dir[0]), float(cur_dir[1]))
            self._trim_points(state, keep_last=self._keep_tail_points)
            return 'maintain_turn'
        blended = 0.75 * prev_dir + 0.25 * cur_dir
        norm    = float(np.linalg.norm(blended))
        if norm > 1e-6:
            blended = blended / norm
            state.segment_dir = (float(blended[0]), float(blended[1]))
        return 'maintain'

    def _evaluate_probation_and_maintain(
        self, state: TrackState, c: dict,
        full_stats: Optional[dict], recent_stats: Optional[dict], ts: int
    ) -> dict:
        recent_total_disp    = float(recent_stats['total_disp'])       if recent_stats else 0.0
        recent_valid_ratio   = float(recent_stats['valid_step_ratio']) if recent_stats else 0.0
        recent_valid_steps   = int(recent_stats['valid_step_count'])   if recent_stats else 0
        recent_path_over_net = float(recent_stats['path_over_net'])    if recent_stats else 999.0
        recent_max_offset    = float(recent_stats['max_offset'])       if recent_stats else 0.0
        recent_dir_ratio     = float(recent_stats['direction_ok_ratio']) if recent_stats else 0.0
        recent_same_sign     = float(recent_stats['same_sign_ratio'])    if recent_stats else 0.0
        recent_sign_flip     = float(recent_stats.get('sign_flip_ratio', 1.0)) if recent_stats else 1.0
        recent_speed_px_ms   = float(self._estimate_recent_speed_px_per_ms(state))

        # P5-G1: 高速且几何质量仍然很干净时，bbox 宽高抖动不再直接 kill。
        # 真实子弹在撞击、拖影、靠近人体边缘时，spatter bbox 会明显变大/变形；
        # 但中心点 recent 运动仍然直、快、单向。此时 bbox 只作为弱证据。
        recent_linear_fast_ok = bool(
            recent_valid_steps >= 2
            and recent_total_disp >= max(12.0, self.min_total_displacement_px * 0.55)
            and recent_path_over_net <= 1.35
            and recent_max_offset <= max(self.maintain_recent_offset_px * 1.25, self.max_line_offset_px * 1.20)
            and recent_dir_ratio >= 0.78
            and recent_same_sign >= 0.78
            and recent_sign_flip <= 0.30
            and recent_speed_px_ms >= max(0.45, self.trigger_min_avg_speed_px_per_ms * 0.75)
        )

        # P10-2: 模型坐标主导时，大残差不再第一帧直接 kill。
        # 先进入 impact_candidate / occlusion_predict 短状态机：
        #   - 第 1 帧 outlier：记录 impact 点，继续保持 1 帧；
        #   - 第 2 帧 outlier：仍允许极短预测，给反弹/遮挡恢复机会；
        #   - 超过 grace 或时间上限：终止在 impact 点，且不允许 same-track 立刻粘回人体框。
        # 这样既保留撞击边界信息，又不会恢复之前不自然的飞行残留。
        if (self.model_track_enabled and self.model_outlier_kill_enabled
                and state.model_outlier_count >= self.model_outlier_kill_frames):
            if state.impact_state == '':
                state.impact_state = 'impact_candidate'
                state.impact_start_ts = int(ts)
                state.impact_outlier_frames = 1
                state.impact_last_ts = int(ts)
                if state.impact_x == 0.0 and state.impact_y == 0.0:
                    state.impact_x = float(state.model_pred_x if state.model_ready else c.get('cx', 0.0))
                    state.impact_y = float(state.model_pred_y if state.model_ready else c.get('cy', 0.0))
                # P10-4: impact 第一帧即把当前可信段入池，供后续新 raw_id 做局部反弹续接。
                self._push_recent_segment(state, 'impact_candidate', int(ts))
                # P12: 若该 bullet 已经正式确认，额外进入所有权池。
                self._push_occlusion_owner(state, 'impact_candidate', int(ts))
            else:
                state.impact_state = 'occlusion_predict'
                state.impact_outlier_frames += 1
                state.impact_last_ts = int(ts)

            impact_age_us = (int(ts) - int(state.impact_start_ts)) if state.impact_start_ts >= 0 else 0
            allow_short_hold = bool(
                state.impact_outlier_frames <= self._impact_outlier_grace_frames
                and impact_age_us <= self._impact_max_hold_us
            )
            if allow_short_hold:
                # P12-Fix：impact/occlusion hold 阶段很容易对应人体残留。
                # 不在 hold 期间提前 display_commit，避免黄色框停在人身上；
                # 真正反弹/穿越段必须由后续新段自证稳定后再继承显示。
                state.p12_polluted_tail = True
                return {
                    'keep_now': True, 'trigger_ok': False, 'maintain_ok': True,
                    'confirmed_now': False,
                    'phase': 'impact_candidate' if state.impact_state == 'impact_candidate' else 'impact_occlusion',
                    'reject_reason': 'model_outlier_hold',
                    'line_ok': True, 'direction_ok': True, 'motion_ok': True,
                    'recent_geom_ok': True, 'recent_motion_ok': True,
                    'trigger_disp_thresh': float(self.min_total_displacement_px),
                    'compact_ok': True,
                }

            # 终止时固定在第一可信 impact 点，避免最后一帧离群 bbox / 预测点把命中点拖远。
            if state.impact_x != 0.0 or state.impact_y != 0.0:
                state.model_x = float(state.impact_x)
                state.model_y = float(state.impact_y)
                state.model_points.append((float(state.impact_x), float(state.impact_y), int(ts)))
            # P12-Fix：终止前不再因为 impact/model_outlier 提前显示，避免把人体残留写入 effect。
            state.p12_polluted_tail = True
            # P12: 终止前把已确认 bullet 的 impact 点锁为 owner，后续新段先尝试继承它。
            self._push_occlusion_owner(state, 'model_outlier', int(ts))
            self._deactivate_bullet(state, reason='model_outlier')
            # P10-7B：保留 last_bullet_id 给 post_outlier_rearm 专用路径。
            # 普通 same-track 已在 _try_reuse_same_track_bullet 中屏蔽 model_outlier，
            # 因此不会立刻粘回人体框。
            return {
                'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                'confirmed_now': False, 'phase': 'terminated_model_outlier',
                'reject_reason': 'model_outlier',
                'line_ok': True, 'direction_ok': False, 'motion_ok': False,
                'recent_geom_ok': False, 'recent_motion_ok': False,
                'trigger_disp_thresh': float(self.min_total_displacement_px),
                'compact_ok': False,
            }

        # P10-2: 如果上一帧处于 impact/occlusion，但本帧观测重新回到模型附近，
        # 说明可能是短遮挡/短暂连框，恢复正常 maintain。
        if state.impact_state and not state.model_outlier:
            state.impact_state = ''
            state.impact_start_ts = -1
            state.impact_outlier_frames = 0
            state.impact_last_ts = -1

        recent_motion_ok = bool(
            recent_total_disp  >= self._recent_static_disp_px
            or recent_valid_steps  >= 1
            or recent_valid_ratio  >= 0.20
        )
        recent_geom_ok = self._size_ok_for_maintain(state, c)

        post_assign_stats = self._post_assign_stats(state)
        post_assign_extra_disp = float(post_assign_stats['total_disp']) if post_assign_stats else 0.0
        post_assign_extra_valid_steps = int(post_assign_stats['valid_step_count']) if post_assign_stats else 0
        probation_new_motion_ok = True
        if state.birth_assign_kind == 'raw_id' and not state.probation_passed:
            probation_new_motion_ok = bool(
                post_assign_extra_disp >= self.probation_new_min_extra_disp_px
                and post_assign_extra_valid_steps >= self.probation_new_min_extra_valid_steps
            )

        # P5-G2 / P10-3: same_track / ghost / ballistic_rearm 续接必须二次确认。
        #
        # 原 P5-G2 只允许严格二次确认：出生后必须新增 >=10px 位移且 >=2 个有效步。
        # 这能防止人体小框粘回旧 bullet，但会误杀 raw87 / raw146 / raw183 这类
        # “反弹后很短、但最近短窗仍然像子弹”的真实候选。
        #
        # P10-3 增加一条“反弹候选 probation”旁路：
        # - 只对 ballistic_rearm / ghost / same_track 等有上下文的候选启用；
        # - 不降低 trigger 主路径；
        # - 必须满足最近短窗速度、直线性、方向、bbox 尺寸保护；
        # - 通过后才允许 probation 继续/通过，仍然会被 model_outlier / bbox / overturn 保护限制。
        reuse_birth_kinds = set(getattr(self, '_reuse_probation_birth_kinds', {
            'ballistic_rearm', 'post_outlier_rearm', 'same_track', 'ghost', 'ghost_capture', 'ghost_capture_bounce', 'segment_bounce', 'segment_forward'
        }))
        is_reuse_probation = (state.birth_assign_kind in reuse_birth_kinds and not state.probation_passed)
        probation_reuse_motion_ok = True
        probation_reuse_strict_ok = True
        probation_reuse_soft_ok = False
        probation_reuse_soft_hold = False
        if is_reuse_probation:
            probation_reuse_strict_ok = bool(
                post_assign_extra_disp >= max(10.0, self.probation_new_min_extra_disp_px)
                and post_assign_extra_valid_steps >= max(2, self.probation_new_min_extra_valid_steps)
                and recent_path_over_net <= 2.20
                and recent_max_offset <= max(self.probation_max_offset_px, self.max_line_offset_px * 1.20)
                and recent_dir_ratio >= 0.70
                and recent_same_sign >= 0.70
                and recent_sign_flip <= 0.35
            )

            cur_w_for_reuse = max(1.0, float(c.get('width', 1.0)))
            cur_h_for_reuse = max(1.0, float(c.get('height', 1.0)))
            cur_area_for_reuse = cur_w_for_reuse * cur_h_for_reuse
            cur_long_for_reuse = max(cur_w_for_reuse, cur_h_for_reuse)
            reuse_bbox_ok = bool(
                cur_area_for_reuse <= self._reuse_probation_max_bbox_area
                and cur_long_for_reuse <= self._reuse_probation_max_bbox_long_side
            )
            probation_reuse_soft_ok = bool(
                reuse_bbox_ok
                and recent_total_disp >= self._reuse_probation_min_disp_px
                and recent_valid_steps >= 1
                and recent_speed_px_ms >= self._reuse_probation_min_speed_px_ms
                and recent_path_over_net <= self._reuse_probation_max_path_over_net
                and recent_max_offset <= self._reuse_probation_max_offset_px
                and recent_dir_ratio >= self._reuse_probation_min_dir_ratio
                and recent_same_sign >= self._reuse_probation_min_same_sign
                and recent_sign_flip <= self._reuse_probation_max_flip_ratio
            )
            probation_reuse_motion_ok = bool(probation_reuse_strict_ok or probation_reuse_soft_ok)

            elapsed_for_soft = int(ts) - int(state.bullet_assign_ts) if state.bullet_assign_ts >= 0 else 0
            probation_reuse_soft_hold = bool(
                (not probation_reuse_motion_ok)
                and reuse_bbox_ok
                and elapsed_for_soft <= self._reuse_probation_grace_us
                and recent_speed_px_ms >= max(0.30, self._reuse_probation_min_speed_px_ms * 0.75)
                and recent_path_over_net <= self._reuse_probation_max_path_over_net * 1.25
                and recent_max_offset <= self._reuse_probation_max_offset_px * 1.20
            )

        # ================================================================
        # P2-C: 转向过多立即 kill（在所有其他检查之前）
        # ================================================================
        if (state.turn_count > self.maintain_max_turn_count
                or state.cumulative_turn_deg > self.maintain_max_cumulative_turn_deg):
            self._deactivate_bullet(state, reason='overturn')
            # 清空 points 防止残余直线点在下一帧立即触发 reactivate_same
            state.points = deque(maxlen=self.history_len)
            state.pre_confirm_points = deque(maxlen=self.pre_confirm_history_len)
            # 清除 last_bullet_id 防止 same_track 重用（人体误判不应该被续接）
            state.last_bullet_id     = None
            state.last_terminated_ts = -1
            return {
                'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                'confirmed_now': False, 'phase': 'terminated_overturn',
                'reject_reason': 'maintain_overturn',
                'line_ok': True, 'direction_ok': False, 'motion_ok': True,
                'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                'trigger_disp_thresh': float(self.min_total_displacement_px),
                'compact_ok': False,
            }

        # ================================================================
        # P0: full_max_offset 全局硬约束
        # ================================================================
        full_max_offset = float(full_stats['max_offset']) if full_stats else 0.0
        if full_max_offset > self.maintain_max_offset_px:
            self._deactivate_bullet(state, reason='offset_exceed')
            return {
                'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                'confirmed_now': False, 'phase': 'terminated_offset',
                'reject_reason': 'maintain_offset',
                'line_ok': False, 'direction_ok': True, 'motion_ok': True,
                'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                'trigger_disp_thresh': float(self.min_total_displacement_px),
                'compact_ok': False,
            }

        # ================================================================
        # P2-A: recent_max_offset 连续超限快速 kill
        # ================================================================
        if recent_max_offset > self.maintain_recent_offset_px:
            state.recent_offset_exceed_count += 1
        else:
            state.recent_offset_exceed_count = 0

        if state.recent_offset_exceed_count >= self.maintain_recent_offset_exceed_frames:
            self._deactivate_bullet(state, reason='recent_offset_exceed')
            return {
                'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                'confirmed_now': False, 'phase': 'terminated_recent_offset',
                'reject_reason': 'maintain_recent_offset',
                'line_ok': False, 'direction_ok': True, 'motion_ok': True,
                'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                'trigger_disp_thresh': float(self.min_total_displacement_px),
                'compact_ok': False,
            }

        # ================================================================
        # P0: bbox 滑窗 std 检查
        # ================================================================
        if len(state.recent_widths) >= 4:
            ws = list(state.recent_widths)
            hs = list(state.recent_heights)
            w_mean = sum(ws) / len(ws)
            h_mean = sum(hs) / len(hs)
            w_std = (sum((w - w_mean) ** 2 for w in ws) / len(ws)) ** 0.5
            h_std = (sum((h - h_mean) ** 2 for h in hs) / len(hs)) ** 0.5
            if w_std > self.maintain_max_bbox_std or h_std > self.maintain_max_bbox_std:
                if not recent_linear_fast_ok:
                    self._deactivate_bullet(state, reason='bbox_unstable')
                    return {
                        'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                        'confirmed_now': False, 'phase': 'terminated_bbox',
                        'reject_reason': 'maintain_bbox',
                        'line_ok': True, 'direction_ok': True, 'motion_ok': True,
                        'recent_geom_ok': False, 'recent_motion_ok': recent_motion_ok,
                        'trigger_disp_thresh': float(self.min_total_displacement_px),
                        'compact_ok': False,
                    }
                # bbox 抖动但 recent 中心轨迹仍然像高速直线：保留 bullet，避免真子弹断线。
                recent_geom_ok = True

        # ================================================================
        # P2-B: bbox EMA 慢漂移检测
        # ================================================================
        cur_w = float(c['width'])
        cur_h = float(c['height'])
        alpha = self.maintain_bbox_ema_alpha
        if state.bbox_ema_w < 0:
            state.bbox_ema_w = cur_w
            state.bbox_ema_h = cur_h
        else:
            state.bbox_ema_w = alpha * cur_w + (1.0 - alpha) * state.bbox_ema_w
            state.bbox_ema_h = alpha * cur_h + (1.0 - alpha) * state.bbox_ema_h
            drift_w = abs(cur_w - state.bbox_ema_w) / max(state.bbox_ema_w, 1.0)
            drift_h = abs(cur_h - state.bbox_ema_h) / max(state.bbox_ema_h, 1.0)
            if drift_w > self.maintain_bbox_ema_drift_ratio or drift_h > self.maintain_bbox_ema_drift_ratio:
                if not recent_linear_fast_ok:
                    self._deactivate_bullet(state, reason='bbox_ema_drift')
                    return {
                        'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                        'confirmed_now': False, 'phase': 'terminated_bbox_ema',
                        'reject_reason': 'maintain_bbox_ema',
                        'line_ok': True, 'direction_ok': True, 'motion_ok': True,
                        'recent_geom_ok': False, 'recent_motion_ok': recent_motion_ok,
                        'trigger_disp_thresh': float(self.min_total_displacement_px),
                        'compact_ok': False,
                    }
                # EMA 漂移但中心轨迹质量高：认为是高速物体 bbox 变形，不立即 kill。
                recent_geom_ok = True

        # ================================================================
        # --- probation 阶段 ---
        # ================================================================
        if state.bullet_active and not state.probation_passed and state.bullet_assign_ts >= 0:
            elapsed          = int(ts) - int(state.bullet_assign_ts)
            disp_since_assign = 0.0
            if state.probation_ref_pos is not None:
                cur_xy_for_probation = (
                    np.array([float(state.model_x), float(state.model_y)], dtype=np.float32)
                    if state.model_ready else
                    np.array([float(c['cx']), float(c['cy'])], dtype=np.float32)
                )
                disp_since_assign = float(np.linalg.norm(
                    cur_xy_for_probation - np.array(state.probation_ref_pos, dtype=np.float32)
                ))

            probation_offset_ok = full_max_offset <= self.probation_max_offset_px

            # P10-3: 对 rearm / ghost 出生候选，若最近短窗已经足够像子弹，
            # 不强制要求出生后立刻累计完整 _probation_continue_disp_px。
            probation_continue_ok = bool(disp_since_assign >= self._probation_continue_disp_px)
            if is_reuse_probation and probation_reuse_soft_ok:
                probation_continue_ok = True

            probation_ok = bool(
                probation_continue_ok
                and recent_valid_steps >= self._probation_recent_valid_steps
                and recent_path_over_net <= self._probation_max_path_over_net
                and probation_offset_ok
                and probation_new_motion_ok
                and probation_reuse_motion_ok
            )
            if probation_ok:
                state.probation_passed     = True
                state.probation_fail_count = 0
                state.static_fail_count    = 0
                state.probation_start_point_count = 0
                state.probation_start_ts = -1
                self._promote_display_id(state)
                phase = self._update_segment_state(state, recent_stats, int(ts))
                return {
                    'keep_now': True, 'trigger_ok': False, 'maintain_ok': True,
                    'confirmed_now': False, 'phase': phase,
                    'reject_reason': 'probation_reuse_soft_pass' if (is_reuse_probation and probation_reuse_soft_ok and not probation_reuse_strict_ok) else 'probation_pass',
                    'line_ok': True, 'direction_ok': True, 'motion_ok': True,
                    'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                    'trigger_disp_thresh': float(self.min_total_displacement_px), 'compact_ok': True,
                }

            # P10-7A：probation 尚未通过，但该段已经有 bullet_id 且短窗质量足够时，
            # 提前进入显示层；不改变 probation 本身是否通过。
            self._try_promote_display_early(state, recent_stats, int(ts))

            if not probation_offset_ok:
                old_bullet_id = state.bullet_id
                self._deactivate_bullet(state, reason='probation_offset')
                return {
                    'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                    'confirmed_now': False, 'phase': 'probation_revoke',
                    'reject_reason': 'probation_offset',
                    'line_ok': False, 'direction_ok': True, 'motion_ok': False,
                    'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                    'trigger_disp_thresh': float(self.min_total_displacement_px),
                    'compact_ok': False,
                    'revoked_bullet_id': int(old_bullet_id) if old_bullet_id is not None else -1,
                }

            if (not probation_new_motion_ok) and elapsed >= self._probation_min_elapsed_us:
                old_bullet_id = state.bullet_id
                self._deactivate_bullet(state, reason='probation_new_motion')
                return {
                    'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                    'confirmed_now': False, 'phase': 'probation_revoke',
                    'reject_reason': 'probation_new_motion',
                    'line_ok': True, 'direction_ok': True, 'motion_ok': False,
                    'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                    'trigger_disp_thresh': float(self.min_total_displacement_px),
                    'compact_ok': False,
                    'revoked_bullet_id': int(old_bullet_id) if old_bullet_id is not None else -1,
                }

            if (not probation_reuse_motion_ok) and elapsed >= self._probation_min_elapsed_us:
                # P10-3: rearm/ghost 反弹候选短暂不稳定时，最多给到 _reuse_probation_grace_us，
                # 但只在 bbox 不大、最近轨迹没有明显人体残留特征时 hold。
                if is_reuse_probation and probation_reuse_soft_hold:
                    return {
                        'keep_now': True, 'trigger_ok': False, 'maintain_ok': True,
                        'confirmed_now': False, 'phase': 'probation_reuse_hold',
                        'reject_reason': 'probation_reuse_soft_hold',
                        'line_ok': True, 'direction_ok': True, 'motion_ok': True,
                        'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                        'trigger_disp_thresh': float(self.min_total_displacement_px),
                        'compact_ok': True,
                        'probation_reuse_soft_ok': int(probation_reuse_soft_ok),
                        'probation_reuse_strict_ok': int(probation_reuse_strict_ok),
                    }

                old_bullet_id = state.bullet_id
                old_stable_id = int(state.stable_track_id)
                self._deactivate_bullet(state, reason='probation_reuse_motion')
                # 失败的续接不应污染 recent_terminated，也不允许继续 same_track 复活。
                self._recent_terminated = [
                    item for item in self._recent_terminated
                    if not (int(item.get('stable_track_id', -1)) == old_stable_id
                            and int(item.get('bullet_id', -9999)) == int(old_bullet_id if old_bullet_id is not None else -9999))
                ]
                state.last_bullet_id = None
                state.last_terminated_ts = -1
                state.last_terminated_pos = None
                state.last_terminated_dir = None
                return {
                    'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                    'confirmed_now': False, 'phase': 'probation_revoke',
                    'reject_reason': 'probation_reuse_motion',
                    'line_ok': True, 'direction_ok': False, 'motion_ok': False,
                    'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                    'trigger_disp_thresh': float(self.min_total_displacement_px),
                    'compact_ok': False,
                    'revoked_bullet_id': int(old_bullet_id) if old_bullet_id is not None else -1,
                    'probation_reuse_soft_ok': int(probation_reuse_soft_ok),
                    'probation_reuse_strict_ok': int(probation_reuse_strict_ok),
                }

            if elapsed >= self._probation_min_elapsed_us:
                state.probation_fail_count += 1

            if elapsed >= self._probation_window_us or state.probation_fail_count >= self._probation_fail_grace:
                old_bullet_id = state.bullet_id
                self._deactivate_bullet(state, reason='probation')
                return {
                    'keep_now': False, 'trigger_ok': False, 'maintain_ok': False,
                    'confirmed_now': False, 'phase': 'probation_revoke',
                    'reject_reason': 'probation_reuse_motion' if not probation_reuse_motion_ok else ('probation_new_motion' if not probation_new_motion_ok else 'probation'),
                    'line_ok': True, 'direction_ok': True, 'motion_ok': False,
                    'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                    'trigger_disp_thresh': float(self.min_total_displacement_px),
                    'compact_ok': False,
                    'revoked_bullet_id': int(old_bullet_id) if old_bullet_id is not None else -1,
                }

            return {
                'keep_now': True, 'trigger_ok': False, 'maintain_ok': True,
                'confirmed_now': False, 'phase': 'probation', 'reject_reason': 'probation_hold',
                'line_ok': True, 'direction_ok': True, 'motion_ok': True,
                'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
                'trigger_disp_thresh': float(self.min_total_displacement_px), 'compact_ok': True,
            }

        # ================================================================
        # --- 维持态（已通过 probation）---
        # ================================================================
        if recent_total_disp <= self._recent_static_disp_px and recent_valid_steps == 0:
            state.static_fail_count += 1
        elif (full_stats is not None
              and full_stats['total_disp'] <= self._long_static_disp_px
              and recent_valid_steps == 0):
            state.static_fail_count += 1
        else:
            state.static_fail_count = 0

        maintain_ok = bool(
            recent_geom_ok and state.static_fail_count < self._stationary_terminate_frames
        )
        if maintain_ok:
            phase         = self._update_segment_state(state, recent_stats, int(ts))
            reject_reason = 'hold_identity'
            keep_now      = True
        else:
            keep_now      = False
            phase         = 'terminated_static'
            reject_reason = 'maintain_static'
            self._deactivate_bullet(state, reason='static')

        return {
            'keep_now': keep_now, 'trigger_ok': False, 'maintain_ok': maintain_ok,
            'confirmed_now': False, 'phase': phase, 'reject_reason': reject_reason,
            'line_ok': True, 'direction_ok': True, 'motion_ok': True,
            'recent_geom_ok': recent_geom_ok, 'recent_motion_ok': recent_motion_ok,
            'trigger_disp_thresh': float(self.min_total_displacement_px), 'compact_ok': True,
        }


    # -----------------------------------------------------------------------
    # P8: 运动模型层。bbox 中心只是 obs，model_x/model_y 才是子弹轨迹坐标。
    # -----------------------------------------------------------------------

    def _reset_model(self, state: TrackState) -> None:
        state.model_ready = False
        state.model_last_ts = -1
        state.model_x = state.model_y = 0.0
        state.model_vx = state.model_vy = 0.0
        state.model_pred_x = state.model_pred_y = 0.0
        state.model_residual_px = -1.0
        state.model_obs_used = False
        state.model_outlier = False
        state.model_outlier_count = 0
        state.model_update_mode = 'reset'
        state.model_speed_px_ms = 0.0
        state.impact_candidate = False
        state.impact_x = state.impact_y = 0.0
        state.impact_state = ''
        state.impact_start_ts = -1
        state.impact_outlier_frames = 0
        state.impact_last_ts = -1
        state.segment_pool_last_push_ts = -1
        state.confirmed_backfill_last_emit_ts = -1
        state.confirmed_backfill_points.clear()
        state.model_points.clear()

    def _seed_model_from_obs_history(self, state: TrackState, ts: int) -> None:
        """P8: bullet 刚触发/续接时，用最近观测初始化运动模型。"""
        obs = [(float(x), float(y), int(t)) for x, y, t in list(state.pre_confirm_points)[-10:]]
        if len(obs) < 2:
            pts = list(state.points)[-6:]
            obs = [(float(p[0]), float(p[1]), int(p[4])) for p in pts]
        # 去重时间戳
        dedup = []
        seen = set()
        for x, y, t in sorted(obs, key=lambda p: p[2]):
            if int(t) in seen:
                continue
            seen.add(int(t))
            dedup.append((x, y, int(t)))
        obs = dedup
        if not obs:
            c = state.last_cluster or {'cx': 0.0, 'cy': 0.0}
            obs = [(float(c.get('cx', 0.0)), float(c.get('cy', 0.0)), int(ts))]

        last_x, last_y, last_ts = obs[-1]
        vx = vy = 0.0
        # 用最近跨度足够的两点估计速度；比单帧中心点更稳。
        for px, py, pts_ts in reversed(obs[:-1]):
            dt_ms = (int(last_ts) - int(pts_ts)) / 1000.0
            dist = math.hypot(last_x - px, last_y - py)
            if dt_ms >= 3.0 and dist >= max(self.min_step_px, 2.5):
                vx = (last_x - px) / max(dt_ms, 1e-6)
                vy = (last_y - py) / max(dt_ms, 1e-6)
                break
        speed = math.hypot(vx, vy)
        if speed < self.model_min_init_speed_px_ms and len(obs) >= 2:
            # 兜底：用首尾估计，避免初始化成静止模型。
            first_x, first_y, first_ts = obs[0]
            dt_ms = (int(last_ts) - int(first_ts)) / 1000.0
            if dt_ms > 1e-6:
                vx = (last_x - first_x) / dt_ms
                vy = (last_y - first_y) / dt_ms
                speed = math.hypot(vx, vy)

        state.model_x = float(last_x)
        state.model_y = float(last_y)
        state.model_vx = float(vx)
        state.model_vy = float(vy)
        state.model_last_ts = int(last_ts)
        state.model_pred_x = float(last_x)
        state.model_pred_y = float(last_y)
        state.model_residual_px = 0.0
        state.model_obs_used = True
        state.model_outlier = False
        state.model_outlier_count = 0
        state.model_update_mode = 'seed'
        state.model_speed_px_ms = float(math.hypot(vx, vy))
        state.model_ready = True
        state.impact_candidate = False
        state.impact_x = float(last_x)
        state.impact_y = float(last_y)
        state.impact_state = ''
        state.impact_start_ts = -1
        state.impact_outlier_frames = 0
        state.impact_last_ts = -1
        state.segment_pool_last_push_ts = -1
        state.model_points.clear()
        for x, y, t in obs[-self.draw_history_len:]:
            state.model_points.append((float(x), float(y), int(t)))

    def _predict_model_xy(self, state: TrackState, ts: int) -> Tuple[float, float]:
        if not state.model_ready or state.model_last_ts < 0:
            c = state.last_cluster or {'cx': 0.0, 'cy': 0.0}
            return float(c.get('cx', 0.0)), float(c.get('cy', 0.0))
        dt_ms = max(0.0, (int(ts) - int(state.model_last_ts)) / 1000.0)
        return (
            float(state.model_x + state.model_vx * dt_ms),
            float(state.model_y + state.model_vy * dt_ms),
        )

    def _update_model_with_observation(self, state: TrackState, c: dict, ts: int, force_seed: bool = False) -> None:
        """P8: 用运动模型吸收观测。离群 bbox 中心不再拉弯轨迹。"""
        obs_x = float(c.get('cx', 0.0))
        obs_y = float(c.get('cy', 0.0))
        ts_i = int(ts)
        state.obs_points.append((obs_x, obs_y, ts_i))

        if (not self.model_track_enabled) or force_seed or (not state.model_ready):
            self._seed_model_from_obs_history(state, ts_i)
            return

        pred_x, pred_y = self._predict_model_xy(state, ts_i)
        residual = float(math.hypot(obs_x - pred_x, obs_y - pred_y))
        speed = float(math.hypot(state.model_vx, state.model_vy))
        # 高速或间隔偏大时，允许残差略增；但不让人物框把轨迹拖走。
        dt_ms = max(0.0, (ts_i - int(state.model_last_ts)) / 1000.0)
        adaptive_soft = max(self.model_soft_residual_px, self.max_line_offset_px * 2.0)
        adaptive_hard = max(self.model_hard_residual_px, adaptive_soft + speed * min(10.0, dt_ms) * 0.35)
        adaptive_outlier = max(self.model_outlier_residual_px, adaptive_hard + 8.0)

        if residual <= adaptive_soft:
            gain = self.model_correction_gain
            mode = 'normal'
            obs_used = True
            outlier = False
        elif residual <= adaptive_hard and self.model_weak_correction_gain > 0.0:
            # P9: 默认关闭弱校正；只有显式设置 gain>0 时才允许可疑观测轻微拉动模型。
            gain = self.model_weak_correction_gain
            mode = 'weak'
            obs_used = True
            outlier = False
        else:
            gain = 0.0
            mode = 'outlier'
            obs_used = False
            outlier = True

        new_x = pred_x + gain * (obs_x - pred_x)
        new_y = pred_y + gain * (obs_y - pred_y)

        if state.model_last_ts >= 0 and dt_ms > 1e-6:
            raw_vx = (new_x - state.model_x) / dt_ms
            raw_vy = (new_y - state.model_y) / dt_ms
            # 限制加速度，避免单个中心点把速度方向瞬间拉弯。
            max_dv = max(0.08, self.model_max_accel_px_ms2 * dt_ms)
            dvx = max(-max_dv, min(max_dv, raw_vx - state.model_vx))
            dvy = max(-max_dv, min(max_dv, raw_vy - state.model_vy))
            if obs_used:
                state.model_vx = float(state.model_vx + dvx)
                state.model_vy = float(state.model_vy + dvy)
            else:
                # 离群时保持原速度，只按预测位置前进。
                new_x, new_y = pred_x, pred_y
        state.model_x = float(new_x)
        state.model_y = float(new_y)
        state.model_last_ts = ts_i
        state.model_pred_x = float(pred_x)
        state.model_pred_y = float(pred_y)
        state.model_residual_px = float(residual)
        state.model_obs_used = bool(obs_used)
        state.model_outlier = bool(outlier)
        state.model_outlier_count = (state.model_outlier_count + 1) if outlier else 0
        state.model_update_mode = str(mode)
        state.model_speed_px_ms = float(math.hypot(state.model_vx, state.model_vy))
        state.impact_candidate = bool(outlier and residual >= adaptive_hard)
        if state.impact_candidate:
            state.impact_x = float(pred_x)
            state.impact_y = float(pred_y)
        else:
            state.impact_x = float(state.model_x)
            state.impact_y = float(state.model_y)
        state.model_points.append((float(state.model_x), float(state.model_y), ts_i))
        if len(state.model_points) > self.draw_history_len:
            state.model_points = deque(list(state.model_points)[-self.draw_history_len:], maxlen=80)

    def _model_recent_stats(self, state: TrackState, window_points: Optional[int] = None) -> Optional[dict]:
        pts = list(state.model_points)
        if window_points is not None:
            pts = pts[-int(window_points):]
        if len(pts) < 2:
            return None
        pseudo = [(float(x), float(y), 1.0, 1.0, int(t)) for x, y, t in pts]
        return self._slice_stats(pseudo)

    def _model_direction(self, state: TrackState) -> Optional[Tuple[float, float]]:
        if len(state.model_points) >= 2:
            p0 = state.model_points[max(0, len(state.model_points) - 4)]
            p1 = state.model_points[-1]
            vx = float(p1[0] - p0[0])
            vy = float(p1[1] - p0[1])
            n = math.hypot(vx, vy)
            if n > 1e-6:
                return (vx / n, vy / n)
        n = math.hypot(state.model_vx, state.model_vy)
        if n > 1e-6:
            return (state.model_vx / n, state.model_vy / n)
        return None

    def _cleanup_missing(self, seen_track_ids: set) -> None:
        dead_ids = []
        for stable_track_id, state in self._tracks.items():
            if stable_track_id not in seen_track_ids:
                state.miss_count += 1
                if state.bullet_active and state.miss_count > self._terminate_miss_frames:
                    self._deactivate_bullet(state, reason='missing')
                if state.miss_count > self.max_missed_frames:
                    dead_ids.append(stable_track_id)
        for stable_track_id in dead_ids:
            del self._tracks[stable_track_id]
        self._rebuild_raw_map()
        self._purge_recent_terminated(
            0 if not self._tracks else max(
                (int(s.last_cluster.get('ts', 0))
                 for s in self._tracks.values() if s.last_cluster is not None),
                default=0,
            )
        )

    # -----------------------------------------------------------------------
    # 主入口
    # -----------------------------------------------------------------------

    def update(self, clusters_np, ts: int) -> List[dict]:
        self._last_update_ts = int(ts)
        accepted:   List[dict] = []
        debug_rows: List[dict] = []
        clusters = [self._cluster_to_dict(c) for c in clusters_np]
        assignments, assign_kind = self._assign_clusters(clusters) if clusters else ({}, {})
        seen_track_ids: set = set()

        for c in clusters:
            raw_id          = int(c['raw_id'])
            stable_track_id = assignments[raw_id]
            state           = self._tracks[stable_track_id]
            seen_track_ids.add(stable_track_id)

            state.miss_count = 0
            c['ts']          = int(ts)
            prev_cluster     = state.last_cluster
            state.last_cluster   = c
            state.current_raw_id = raw_id
            if state.first_seen_ts < 0:
                state.first_seen_ts = int(ts)
            if not state.raw_id_history or state.raw_id_history[-1] != raw_id:
                state.raw_id_history.append(raw_id)

            # P2-G: 静止点过滤——若新点与上一点距离 < min_step_px * 0.5
            #        则跳过追加到 state.points，避免污染 valid_step_ratio。
            #        pre_confirm_points 和 display_points 不受影响。
            skip_point = False
            if state.points:
                last_pt = state.points[-1]
                dx = c['cx'] - last_pt[0]
                dy = c['cy'] - last_pt[1]
                if (dx * dx + dy * dy) < (self.min_step_px * 0.5) ** 2:
                    skip_point = True
            if not skip_point:
                state.points.append((c['cx'], c['cy'], c['width'], c['height'], int(ts)))

            # bbox EMA 和滑窗 std 的原始数据不受静止点过滤影响，每帧都更新
            state.recent_widths.append(float(c['width']))
            state.recent_heights.append(float(c['height']))
            self._append_pre_confirm_point(state, c['cx'], c['cy'], int(ts))
            state.obs_points.append((float(c['cx']), float(c['cy']), int(ts)))
            if state.bullet_active:
                # P8: 已确认/试用中的子弹先更新模型，再进入 maintain 判定；bbox 中心只作为观测。
                self._update_model_with_observation(state, c, int(ts))
                if len(state.model_points) >= 1:
                    mx, my, mts = state.model_points[-1]
                    self._append_display_point(state, mx, my, int(mts))
                else:
                    self._append_display_point(state, c['cx'], c['cy'], int(ts))

            full_stats, recent_stats = self._compute_stats(state)
            model_full_stats = self._model_recent_stats(state, None) if state.model_ready else None
            model_recent_stats = self._model_recent_stats(state, self._recent_window_points) if state.model_ready else None

            if state.bullet_active:
                # P8: maintain 阶段优先使用模型轨迹统计，避免人物框中心点折线影响判定。
                eval_result = self._evaluate_probation_and_maintain(
                    state, c if prev_cluster is not None else c,
                    model_full_stats if model_full_stats is not None else full_stats,
                    model_recent_stats if model_recent_stats is not None else recent_stats,
                    int(ts)
                )
                trigger_ok = False
            else:
                eval_result = self._evaluate_trigger(
                    state, full_stats, recent_stats,
                    assign_kind.get(raw_id, 'unknown'), int(ts)
                )
                trigger_ok = bool(eval_result['trigger_ok'])

            keep_now = bool(eval_result['keep_now'])
            phase    = str(eval_result['phase'])

            if keep_now:
                state.keep_streak += 1
            else:
                state.keep_streak = 0

            just_assigned_bullet = False
            confirmed_now        = False
            reuse_source         = 'none'
            # P10-5: 当前 cluster 若刚确认 bullet，可在真实 row 前插入历史 backfill rows。
            backfill_rows_to_emit: List[dict] = []

            required_trigger_streak = int(eval_result.get('required_trigger_streak_override', 0) or 0)
            if required_trigger_streak <= 0:
                required_trigger_streak = self._required_trigger_streak(recent_stats)

            if (not state.bullet_active) and trigger_ok and state.keep_streak >= required_trigger_streak:
                is_post_outlier_rearm = bool(eval_result.get('post_outlier_rearm', False))
                near_owner_guard = self._is_near_occlusion_owner(c, int(ts))
                reused_bid = None
                reuse_source = 'none'
                if is_post_outlier_rearm:
                    # P12-Fix: post_outlier_rearm 在 owner 污染窗口内不能直接复活旧 bullet。
                    # 先把点重置到近期后缀，再要求它按 owner_forward/owner_bounce 的严格规则自证。
                    self._reset_points_to_recent_suffix_for_rearm(
                        state, int(eval_result.get('post_outlier_rearm_window', 0) or 0)
                    )
                    if near_owner_guard:
                        reused_bid, reuse_source = self._try_occlusion_owner_capture_new_track(state, c, int(ts))
                    else:
                        reused_bid = int(state.last_bullet_id) if state.last_bullet_id is not None else None
                        reuse_source = 'post_outlier_rearm' if reused_bid is not None else 'new'
                else:
                    # P15: 新 raw_id 已经满足 trigger 时，也必须先尝试继承旧 bullet_id。
                    # 旧版只在 trigger 失败/ghost 分支才走 segment_pool，导致新段一旦很快自证为子弹，
                    # 就直接开新 bullet_id，同一颗子弹被拆成多个 ID。
                    reused_bid, reuse_source = self._try_occlusion_owner_capture_new_track(state, c, int(ts))
                    if reused_bid is None:
                        reused_bid, reuse_source = self._try_segment_pool_capture_new_track(state, c, int(ts))
                    if reused_bid is None and not near_owner_guard:
                        reused_bid, reuse_source = self._try_reuse_same_track_bullet(state, c, int(ts))
                    if reused_bid is None and not near_owner_guard:
                        reused_bid, reuse_source = self._try_reuse_recent_terminated_bullet(state, c, int(ts))
                if reused_bid is None:
                    # P12: 若当前点处于已确认 bullet 的人体/撞击污染保护区，
                    # 但又不能解释为 owner_forward / owner_bounce，则禁止它独立开新 bullet。
                    if self._is_near_occlusion_owner(c, int(ts)):
                        phase = 'p12_owner_residual_guard'
                        eval_result['reject_reason'] = 'p12_owner_residual_guard'
                        keep_now = False
                        state.keep_streak = 0
                        just_assigned_bullet = False
                    else:
                        state.bullet_id  = self._next_bullet_id
                        self._next_bullet_id += 1
                        reuse_source     = 'new'
                else:
                    state.bullet_id  = int(reused_bid)
                if state.bullet_id is None:
                    # P12 residual guard 已经拦截，保持普通 track，不进入 bullet_active。
                    pass
                else:
                    if reuse_source == 'new':
                        state.display_points = deque(maxlen=self.draw_history_len)
                    self._backfill_display_from_preconfirm(state)
                    self._seed_model_from_obs_history(state, int(ts))
                    state.display_points = deque(list(state.model_points), maxlen=self.draw_history_len) if len(state.model_points) >= 2 else state.display_points
                    state.bullet_active         = True
                    state.confirmed_once        = True
                    state.static_fail_count     = 0
                    state.bullet_assign_ts      = int(ts)
                    state.probation_ref_pos     = (float(state.model_x), float(state.model_y)) if state.model_ready else (float(c['cx']), float(c['cy']))
                    state.probation_fail_count  = 0
                    # P5-G2: 即使是 same_track/ghost 复用旧 bullet_id，也先进入短 probation，
                    # 防止撞击人体后把人体小框直接续成原子弹。
                    state.probation_passed      = False
                    state.probation_until_ts    = int(ts) + self._probation_window_us
                    state.probation_start_point_count = len(state.points)
                    state.probation_start_ts    = int(ts)
                    if bool(eval_result.get('post_outlier_rearm', False)):
                        state.birth_assign_kind = 'post_outlier_rearm'
                    elif reuse_source == 'new' and bool(eval_result.get('ballistic_rearm', False)):
                        state.birth_assign_kind = 'ballistic_rearm'
                    else:
                        state.birth_assign_kind = assign_kind.get(raw_id, 'raw_id') if reuse_source == 'new' else reuse_source
                    state.display_bullet_id     = int(state.bullet_id) if int(state.bullet_id) in self._displayed_bullet_ids else None
                    if state.probation_passed and state.display_bullet_id is None:
                        self._promote_display_id(state)
                    # P10-5: 确认后回填前半段可信候选轨迹。
                    _bf_pts = self._select_confirmed_backfill_points(state, int(ts))
                    if _bf_pts:
                        self._apply_confirmed_backfill_points(state, _bf_pts)
                        # P10-7A：回填点已证明该候选段在确认前就存在，
                        # 因此在生成 backfill debug rows 前先尝试提升显示 ID，
                        # 让 effect CSV 的回填行也能带 display_bullet_id。
                        self._try_promote_display_early(state, recent_stats, int(ts))
                        backfill_rows_to_emit = self._build_confirmed_backfill_debug_rows(
                            state, raw_id, stable_track_id, c, _bf_pts, int(ts)
                        )
                    just_assigned_bullet        = True
                    confirmed_now               = True
                    # P2-C: trigger 时重置转向统计
                    state.turn_count            = 0
                    state.cumulative_turn_deg   = 0.0
                    state.recent_offset_exceed_count = 0
                    if str(getattr(state, 'p12_owner_mode', '')) in {'owner_forward', 'owner_bounce'}:
                        state.segment_index = max(0, int(getattr(state, 'p12_inherited_segment_index', -1)))
                        state.segment_turn_warmup_until_ts = int(ts) + self._segment_turn_warmup_us
                    elif bool(eval_result.get('post_outlier_rearm', False)):
                        state.segment_index = int(state.segment_index) + 1
                        state.post_outlier_rearm_count += 1
                        state.post_outlier_rearm_last_ts = int(ts)
                        state.post_outlier_rearm_last_window = int(eval_result.get('post_outlier_rearm_window', 0) or 0)
                        state.segment_turn_warmup_until_ts = int(ts) + self._segment_turn_warmup_us
                    elif reuse_source in {'ghost_capture_bounce', 'segment_bounce'}:
                        state.segment_index = int(state.segment_index) + 1
                        state.segment_turn_warmup_until_ts = int(ts) + self._segment_turn_warmup_us
                    else:
                        state.segment_index = 0
                    d = self._track_direction(state, window_points=min(4, max(2, len(state.points))))
                    state.segment_dir           = (float(d[0]), float(d[1])) if d is not None else None
                    state.bbox_reuse_min_disp_px = 0.0
                    if reuse_source == 'post_outlier_rearm':
                        phase = 'reactivate_post_outlier_rearm'
                    elif reuse_source == 'same_track':
                        phase = 'reactivate_same'
                    elif reuse_source == 'ghost':
                        phase = 'reactivate_ghost'
                    elif reuse_source in {'owner_forward', 'owner_bounce'}:
                        phase = reuse_source
                    else:
                        phase = 'trigger'


            # ================================================================
            # P1/P2-E: Ghost 跨 track 捕获（拆成直飞+反弹两条路径）
            # ================================================================
            if (not state.bullet_active
                    and not just_assigned_bullet
                    and assign_kind.get(raw_id) == 'new'
                    and len(state.points) <= self._segment_bounce_candidate_max_points):
                # P12-Fix: owner 污染窗口内只允许严格 owner 续接；
                # 不再 fallback 到 segment_pool / ghost，防止人体残留绕过 guard。
                near_owner_guard = self._is_near_occlusion_owner(c, int(ts))
                ghost_bid, ghost_src = self._try_occlusion_owner_capture_new_track(state, c, int(ts))
                if ghost_bid is None and not near_owner_guard:
                    # P10-4: 基于 recent segment pool 的局部续接；失败再走原 ghost/bounce。
                    ghost_bid, ghost_src = self._try_segment_pool_capture_new_track(state, c, int(ts))
                if ghost_bid is None and (not near_owner_guard) and len(state.points) <= 2:
                    ghost_bid, ghost_src = self._try_ghost_capture_new_track(c, int(ts))
                if ghost_bid is None and near_owner_guard:
                    # 处于 owner 污染窗口，但不能解释为穿越/反弹：人体残留，不允许独立显示。
                    phase = 'p12_owner_residual_guard'
                    keep_now = False
                    state.keep_streak = 0
                if ghost_bid is not None:
                    state.bullet_id         = int(ghost_bid)
                    state.display_points    = deque(maxlen=self.draw_history_len)
                    self._backfill_display_from_preconfirm(state)
                    self._seed_model_from_obs_history(state, int(ts))
                    state.display_points = deque(list(state.model_points), maxlen=self.draw_history_len) if len(state.model_points) >= 2 else state.display_points
                    state.bullet_active     = True
                    state.confirmed_once    = True
                    state.static_fail_count = 0
                    state.bullet_assign_ts  = int(ts)
                    state.probation_ref_pos = (float(state.model_x), float(state.model_y)) if state.model_ready else (float(c['cx']), float(c['cy']))
                    state.probation_fail_count = 0
                    # P5-G2: ghost_capture 只是“疑似续接”，不能直接通过 probation。
                    state.probation_passed  = False
                    state.probation_until_ts = int(ts) + self._probation_window_us
                    state.probation_start_point_count = len(state.points)
                    state.probation_start_ts = int(ts)
                    state.birth_assign_kind = ghost_src
                    state.display_bullet_id = int(state.bullet_id) if int(state.bullet_id) in self._displayed_bullet_ids else None
                    # P10-5: ghost/segment 续接确认后，也回填该新段自己已经观测到的可信起始点。
                    _bf_pts = self._select_confirmed_backfill_points(state, int(ts))
                    if _bf_pts:
                        self._apply_confirmed_backfill_points(state, _bf_pts)
                        # P10-7A：回填点已证明该候选段在确认前就存在，
                        # 因此在生成 backfill debug rows 前先尝试提升显示 ID。
                        self._try_promote_display_early(state, recent_stats, int(ts))
                        backfill_rows_to_emit = self._build_confirmed_backfill_debug_rows(
                            state, raw_id, stable_track_id, c, _bf_pts, int(ts)
                        )
                    just_assigned_bullet    = True
                    confirmed_now           = True
                    # P2-C: ghost capture 时重置转向统计
                    state.turn_count        = 0
                    state.cumulative_turn_deg = 0.0
                    state.recent_offset_exceed_count = 0
                    if str(getattr(state, 'p12_owner_mode', '')) in {'owner_forward', 'owner_bounce'}:
                        state.segment_index = max(0, int(getattr(state, 'p12_inherited_segment_index', -1)))
                    elif ghost_src in {'ghost_capture_bounce', 'segment_bounce'}:
                        state.segment_index = int(state.segment_index) + 1
                    else:
                        state.segment_index = 0
                    state.segment_dir       = None
                    state.segment_turn_warmup_until_ts = int(ts) + self._segment_turn_warmup_us
                    state.bbox_reuse_min_disp_px = 0.0
                    reuse_source            = ghost_src
                    phase                   = ghost_src
                    keep_now                = True
                    state.keep_streak       = max(state.keep_streak, self.bullet_min_output_streak)

            accepted_now = bool(state.bullet_active)
            # P12-Fix: 已确认 bullet 进入 impact/occlusion 后，当前 bbox 往往来自人体残留。
            # 这类行保留在 debug 中，但不作为正式显示/effect 行输出，避免黄色框停在人身上。
            suppress_display_row = bool(
                getattr(state, 'p12_polluted_tail', False)
                or bool(getattr(state, 'model_outlier', False))
                or bool(str(getattr(state, 'impact_state', '')))
                or str(phase) in {'impact_candidate', 'impact_occlusion', 'terminated_model_outlier', 'p12_owner_residual_guard'}
            )
            effective_display_bullet_id = (
                int(state.display_bullet_id)
                if (state.display_bullet_id is not None and not suppress_display_row) else -1
            )

            if accepted_now:
                out = dict(c)
                # P8: 保留 obs_x/obs_y 为检测框中心；cx/cy 继续代表观测中心，model_x/model_y 是子弹轨迹坐标。
                out['obs_x'] = float(c['cx'])
                out['obs_y'] = float(c['cy'])
                out['model_x'] = float(state.model_x) if state.model_ready else float(c['cx'])
                out['model_y'] = float(state.model_y) if state.model_ready else float(c['cy'])
                out['pred_x'] = float(state.model_pred_x) if state.model_ready else float(c['cx'])
                out['pred_y'] = float(state.model_pred_y) if state.model_ready else float(c['cy'])
                out['model_residual_px'] = float(state.model_residual_px)
                out['model_obs_used'] = int(state.model_obs_used)
                out['model_outlier'] = int(state.model_outlier)
                out['model_update_mode'] = str(state.model_update_mode)
                out['model_speed_px_ms'] = float(state.model_speed_px_ms)
                out['impact_candidate'] = int(state.impact_candidate)
                out['impact_x'] = float(state.impact_x)
                out['impact_y'] = float(state.impact_y)
                out['impact_state'] = str(state.impact_state)
                out['impact_outlier_frames'] = int(state.impact_outlier_frames)
                out['bullet_id']       = int(state.bullet_id)
                out['display_bullet_id'] = int(effective_display_bullet_id)
                out['stable_track_id'] = int(stable_track_id)
                out['confirmed']       = bool(confirmed_now)
                out['phase']           = phase
                out['segment_index']   = int(state.segment_index)
                if full_stats is not None:
                    out.update({
                        'total_disp_px':      float(full_stats['total_disp']),
                        'max_offset_px':      float(full_stats['max_offset']),
                        'mean_offset_px':     float(full_stats['mean_offset']),
                        'direction_ok_ratio': float(full_stats['direction_ok_ratio']),
                        'same_sign_ratio':    float(full_stats['same_sign_ratio']),
                        'valid_step_count':   int(full_stats['valid_step_count']),
                        'valid_step_ratio':   float(full_stats['valid_step_ratio']),
                        'keep_streak':        int(state.keep_streak),
                    })
                accepted.append(out)

            # P10-5: backfill rows 必须排在当前真实 row 前面，
            # 这样 BulletEventExtractor 的 approach_point 历史能先拿到确认前轨迹。
            if backfill_rows_to_emit:
                debug_rows.extend(backfill_rows_to_emit)

            track_age_ms = (int(ts) - int(state.first_seen_ts)) / 1000.0 if state.first_seen_ts >= 0 else -1.0
            row = {
                'timestamp': int(ts), 'raw_id': int(raw_id),
                'stable_track_id': int(stable_track_id),
                'assign_kind': assign_kind.get(raw_id, 'unknown'),
                'x': float(c['x']), 'y': float(c['y']),
                'width': float(c['width']), 'height': float(c['height']),
                'cx': float(c['cx']), 'cy': float(c['cy']),
                # P8: obs 是检测框中心；model 是真实子弹轨迹坐标，后续命中事件优先使用 model。
                'obs_x': float(c['cx']), 'obs_y': float(c['cy']),
                'model_x': float(state.model_x) if state.model_ready else float(c['cx']),
                'model_y': float(state.model_y) if state.model_ready else float(c['cy']),
                'pred_x': float(state.model_pred_x) if state.model_ready else float(c['cx']),
                'pred_y': float(state.model_pred_y) if state.model_ready else float(c['cy']),
                'model_residual_px': float(state.model_residual_px),
                'model_obs_used': int(state.model_obs_used),
                'model_outlier': int(state.model_outlier),
                'model_update_mode': str(state.model_update_mode),
                'model_speed_px_ms': float(state.model_speed_px_ms),
                'impact_candidate': int(state.impact_candidate),
                'impact_x': float(state.impact_x), 'impact_y': float(state.impact_y),
                'impact_state': str(state.impact_state),
                'impact_outlier_frames': int(state.impact_outlier_frames),
                'n_points': len(state.points), 'model_n_points': len(state.model_points), 'miss_count': int(state.miss_count),
                'keep_now': int(keep_now), 'confirmed_now': int(confirmed_now),
                'accepted_now': int(accepted_now),
                'bullet_id': int(state.bullet_id) if state.bullet_id is not None else -1,
                'display_bullet_id': int(effective_display_bullet_id),
                'just_assigned_bullet': int(just_assigned_bullet),
                'birth_assign_kind': state.birth_assign_kind,
                'p12_owner_mode': str(getattr(state, 'p12_owner_mode', '')),
                'p12_owner_bullet_id': int(getattr(state, 'p12_owner_bullet_id', -1)),
                'p12_inherited_segment_index': int(getattr(state, 'p12_inherited_segment_index', -1)),
                'p12_owner_reason': str(getattr(state, 'p12_owner_reason', '')),
                'p12_owner_pending': int(bool(getattr(state, 'p12_owner_pending', False))),
                'p12_owner_pending_mode': str(getattr(state, 'p12_owner_pending_mode', '')),
                'p12_polluted_tail': int(bool(getattr(state, 'p12_polluted_tail', False))),
                'keep_streak_after': int(state.keep_streak),
                'reject_reason': str(eval_result['reject_reason']),
                'line_ok': int(eval_result['line_ok']),
                'direction_ok': int(eval_result['direction_ok']),
                'motion_ok': int(eval_result['motion_ok']),
                'total_disp_px':       float(full_stats['total_disp'])           if full_stats  else -1.0,
                'max_offset_px':       float(full_stats['max_offset'])           if full_stats  else -1.0,
                'mean_offset_px':      float(full_stats['mean_offset'])          if full_stats  else -1.0,
                'direction_ok_ratio':  float(full_stats['direction_ok_ratio'])   if full_stats  else -1.0,
                'same_sign_ratio':     float(full_stats['same_sign_ratio'])      if full_stats  else -1.0,
                'valid_step_count':    int(full_stats['valid_step_count'])        if full_stats  else -1,
                'valid_step_ratio':    float(full_stats['valid_step_ratio'])      if full_stats  else -1.0,
                'total_step_count':    int(full_stats['total_step_count'])        if full_stats  else -1,
                'phase': phase, 'trigger_ok': int(eval_result['trigger_ok']),
                'maintain_ok': int(eval_result['maintain_ok']),
                'bullet_active': int(state.bullet_active),
                'last_bullet_id': int(state.last_bullet_id) if state.last_bullet_id is not None else -1,
                'static_fail_count': int(state.static_fail_count),
                'trigger_disp_thresh': float(eval_result['trigger_disp_thresh']),
                'recent_total_disp_px':      float(recent_stats['total_disp'])          if recent_stats else -1.0,
                'recent_max_offset_px':      float(recent_stats['max_offset'])          if recent_stats else -1.0,
                'recent_mean_offset_px':     float(recent_stats['mean_offset'])         if recent_stats else -1.0,
                'recent_direction_ok_ratio': float(recent_stats['direction_ok_ratio'])  if recent_stats else -1.0,
                'recent_same_sign_ratio':    float(recent_stats['same_sign_ratio'])     if recent_stats else -1.0,
                'recent_valid_step_count':   int(recent_stats['valid_step_count'])       if recent_stats else -1,
                'recent_valid_step_ratio':   float(recent_stats['valid_step_ratio'])     if recent_stats else -1.0,
                'recent_geom_ok':    int(eval_result['recent_geom_ok']),
                'recent_motion_ok':  int(eval_result['recent_motion_ok']),
                # P2-H: 补全调试字段
                'track_age_ms':             float(track_age_ms),
                'path_over_net':            float(full_stats['path_over_net'])        if full_stats    else -1.0,
                'recent_path_over_net':     float(recent_stats['path_over_net'])      if recent_stats  else -1.0,
                'reuse_source':             reuse_source,
                'recent_segment_pool_size': int(len(self._recent_segments)),
                'segment_pool_candidate':   int(1 if str(reuse_source).startswith('segment_') else 0),
                'probation_passed':         int(state.probation_passed),
                'probation_fail_count':     int(state.probation_fail_count),
                'segment_index':            int(state.segment_index),
                'compact_ok':               int(eval_result.get('compact_ok', False)),
                # P10-1: recent ballistic rearm 调试字段
                'rearm_mode':               int(eval_result.get('rearm_mode', False)),
                'ballistic_rearm':          int(eval_result.get('ballistic_rearm', False)),
                'ballistic_rearm_window':   int(eval_result.get('ballistic_rearm_window', 0)),
                'ballistic_rearm_speed_px_ms': float(eval_result.get('ballistic_rearm_speed_px_ms', 0.0)),
                # P10-6: post-outlier rearm 调试字段。
                'post_outlier_rearm':        int(eval_result.get('post_outlier_rearm', False)),
                'post_outlier_rearm_window': int(eval_result.get('post_outlier_rearm_window', 0)),
                'post_outlier_rearm_dt_us':  int(eval_result.get('post_outlier_rearm_dt_us', 0)),
                'post_outlier_rearm_count':  int(state.post_outlier_rearm_count),
                'required_trigger_streak':  int(required_trigger_streak),
                # P10-3: rearm/ghost probation 调试字段。
                'probation_reuse_soft_ok':   int(eval_result.get('probation_reuse_soft_ok', 0)),
                'probation_reuse_strict_ok': int(eval_result.get('probation_reuse_strict_ok', 0)),
                'turn_count':               int(state.turn_count),
                'cumulative_turn_deg':      float(state.cumulative_turn_deg),
                'recent_offset_exceed_count': int(state.recent_offset_exceed_count),
                # P5: 补充调试字段
                'sign_flip_ratio':          float(full_stats.get('sign_flip_ratio', -1.0))  if full_stats   else -1.0,
                'recent_sign_flip_ratio':   float(recent_stats.get('sign_flip_ratio', -1.0)) if recent_stats else -1.0,
                # P10-5: 确认后候选轨迹回填诊断。旧主脚本 CSV 表头会忽略此字段，但 debug_rows 内部可用。
                'confirmed_backfill': 0,
                'confirmed_backfill_n_points': int(len(state.confirmed_backfill_points)),
            }
            debug_rows.append(row)

        self._cleanup_missing(seen_track_ids)
        self._last_debug_rows = debug_rows
        return accepted

    def get_last_debug_rows(self) -> List[dict]:
        return list(self._last_debug_rows)

    def _filter_display_spikes(self, pts_src: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """P6: 只用于绘制的单点尖刺过滤。

        当某一帧 spatter 框被人物/撞击碎片拉偏时，轨迹会出现一个很尖的 V 形折线，
        但前后点整体仍在原方向上。这里删除这种单点视觉尖刺，不改变 bullet_id 判定。
        P7 的运动学拟合会在此基础上进一步把轨迹画成平滑物理曲线。
        """
        if (not self._draw_spike_filter_enabled) or len(pts_src) < 3:
            return pts_src
        out: List[Tuple[float, float, int]] = []
        n = len(pts_src)
        for i, p in enumerate(pts_src):
            if i == 0 or i == n - 1:
                out.append(p)
                continue
            p0 = np.array([pts_src[i - 1][0], pts_src[i - 1][1]], dtype=np.float32)
            p1 = np.array([p[0], p[1]], dtype=np.float32)
            p2 = np.array([pts_src[i + 1][0], pts_src[i + 1][1]], dtype=np.float32)
            v01 = p1 - p0
            v12 = p2 - p1
            v02 = p2 - p0
            d01 = float(np.linalg.norm(v01))
            d12 = float(np.linalg.norm(v12))
            d02 = float(np.linalg.norm(v02))
            if d01 < self._draw_spike_min_leg_px or d12 < self._draw_spike_min_leg_px or d02 < 1e-6:
                out.append(p)
                continue
            cross = abs(float(v02[0] * (p0[1] - p1[1]) - v02[1] * (p0[0] - p1[0])))
            perp = cross / max(d02, 1e-6)
            cos_turn = float(np.dot(v01, v12) / max(d01 * d12, 1e-6))
            if perp >= self._draw_spike_perp_px and cos_turn <= 0.15:
                continue
            out.append(p)
        return out

    @staticmethod
    def _polyfit_predict_1d(t: np.ndarray, y: np.ndarray, deg: int) -> Tuple[np.ndarray, np.ndarray]:
        """安全的一维 polyfit；返回 coeff, pred。"""
        deg = int(max(1, min(2, deg)))
        if len(t) <= deg:
            deg = max(1, len(t) - 1)
        coeff = np.polyfit(t, y, deg=deg)
        pred = np.polyval(coeff, t)
        return coeff, pred

    def _extend_display_tail_after_terminate(self, pts_src: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """P7: 终止后基于最后速度做极短尾迹外推。

        raw50/raw61/raw74 这类撞击后观测经常已经不再像稳定子弹，不能长期分配 bullet_id，
        但画面上需要看到子弹撞击前的最后运动方向。这里只在绘制 hold 轨迹时外推
        20~35ms 或最多约 55px，不参与判定。
        """
        if (not getattr(self, '_draw_extend_tail_after_terminate', False)) or len(pts_src) < 4:
            return pts_src
        pts = sorted(pts_src, key=lambda p: int(p[2]))
        tail = pts[-min(6, len(pts)):]
        arr = np.array([[float(p[0]), float(p[1]), int(p[2])] for p in tail], dtype=np.float64)
        t = (arr[:, 2] - arr[0, 2]) / 1000.0
        if float(t[-1] - t[0]) < 6.0:
            return pts_src
        try:
            cx, _ = self._polyfit_predict_1d(t, arr[:, 0], 1)
            cy, _ = self._polyfit_predict_1d(t, arr[:, 1], 1)
        except Exception:
            return pts_src
        vx = float(cx[0]) if len(cx) == 2 else 0.0
        vy = float(cy[0]) if len(cy) == 2 else 0.0
        speed = float(math.hypot(vx, vy))
        if speed < max(0.25, self.trigger_min_avg_speed_px_per_ms * 0.45):
            return pts_src
        # 只外推一个很短尾巴，防止重新出现“粘在人身上”的长线。
        extend_ms = min(35.0, max(15.0, 55.0 / max(speed, 1e-6)))
        last_t = float(t[-1])
        last_ts = int(arr[-1, 2])
        new_pts = list(pts_src)
        for frac in (0.33, 0.66, 1.0):
            tt = last_t + extend_ms * frac
            xx = float(np.polyval(cx, tt))
            yy = float(np.polyval(cy, tt))
            ts_i = int(round(last_ts + extend_ms * frac * 1000.0))
            new_pts.append((xx, yy, ts_i))
        return new_pts

    def _fit_kinematic_display_path(self, pts_src: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """P7: 用鲁棒运动学模型生成绘制轨迹。

        旧画法是直接连接 bbox 中心点；靠近人物区域时，人的小框和子弹框连在一起，
        bbox 中心会被拉偏，画面就会出现明显折线。

        新画法把 bbox 中心当作“带噪声观测”：
          - 先过滤单点尖刺；
          - 再对 x(t)、y(t) 做鲁棒一次/二次拟合；
          - 残差过大的观测点视为人身噪声/飞溅碎片，不拉动最终黄色轨迹；
          - 二次项受限，避免拟合出不符合物理的夸张弯曲。
        只影响绘制，不改变 raw_id/bullet_id 判定和命中事件。
        """
        pts_src = self._filter_display_spikes(pts_src)
        if (not self._draw_kinematic_fit_enabled
                or len(pts_src) < self._draw_kinematic_min_points):
            return pts_src

        # 按时间排序，同时去掉完全重复的时间戳，避免 polyfit 病态。
        pts_sorted = sorted(pts_src, key=lambda p: int(p[2]))
        dedup: List[Tuple[float, float, int]] = []
        seen_ts = set()
        for x, y, ts in pts_sorted:
            ts_i = int(ts)
            if ts_i in seen_ts:
                continue
            seen_ts.add(ts_i)
            dedup.append((float(x), float(y), ts_i))
        if len(dedup) < self._draw_kinematic_min_points:
            return pts_src

        arr = np.array([[p[0], p[1], p[2]] for p in dedup], dtype=np.float64)
        x = arr[:, 0]
        y = arr[:, 1]
        ts = arr[:, 2]
        t_ms = (ts - ts[0]) / 1000.0
        duration_ms = float(t_ms[-1] - t_ms[0])
        if duration_ms < 6.0:
            return pts_src
        chord_len = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        if chord_len < max(8.0, self.min_total_displacement_px * 0.35):
            return pts_src

        # P9: 子弹在单段内应近似直线；不再默认拟合二次曲线，避免“弯弯绕绕”。
        deg = 1 if self._draw_kinematic_force_straight_line else (2 if (len(dedup) >= self._draw_kinematic_quad_min_points and duration_ms >= 18.0) else 1)
        mask = np.ones(len(dedup), dtype=bool)
        residual_limit = float(self._draw_kinematic_base_residual_px)

        coeff_x = coeff_y = None
        for _ in range(4):
            if int(mask.sum()) <= deg + 1:
                break
            coeff_x, pred_x_fit = self._polyfit_predict_1d(t_ms[mask], x[mask], deg)
            coeff_y, pred_y_fit = self._polyfit_predict_1d(t_ms[mask], y[mask], deg)
            pred_x = np.polyval(coeff_x, t_ms)
            pred_y = np.polyval(coeff_y, t_ms)
            resid = np.hypot(x - pred_x, y - pred_y)
            med = float(np.median(resid[mask])) if int(mask.sum()) else float(np.median(resid))
            mad = float(np.median(np.abs(resid[mask] - med))) if int(mask.sum()) else 0.0
            robust_limit = max(residual_limit, med + 3.0 * max(1.0, 1.4826 * mad))
            new_mask = resid <= robust_limit
            # 至少保留首尾，保证轨迹长度不被中间噪声截短。
            new_mask[0] = True
            new_mask[-1] = True
            if int(new_mask.sum()) < max(3, deg + 2):
                break
            if np.array_equal(new_mask, mask):
                mask = new_mask
                break
            mask = new_mask

        if coeff_x is None or coeff_y is None or int(mask.sum()) < 3:
            return pts_src

        # 如果二次项导致曲线偏离首尾弦线过大，或者加速度不符合当前弹道尺度，则退回一次直线。
        if deg == 2:
            sample_t_chk = np.linspace(float(t_ms[0]), float(t_ms[-1]), 24)
            px_chk = np.polyval(coeff_x, sample_t_chk)
            py_chk = np.polyval(coeff_y, sample_t_chk)
            p0 = np.array([px_chk[0], py_chk[0]], dtype=np.float64)
            p1 = np.array([px_chk[-1], py_chk[-1]], dtype=np.float64)
            v = p1 - p0
            vlen = float(np.linalg.norm(v))
            max_curve = 0.0
            if vlen > 1e-6:
                vv = v / vlen
                for xx, yy in zip(px_chk, py_chk):
                    pp = np.array([xx, yy], dtype=np.float64)
                    proj = p0 + vv * float(np.dot(pp - p0, vv))
                    max_curve = max(max_curve, float(np.linalg.norm(pp - proj)))
            accel = 0.0
            try:
                # x = ax*t^2 + bx*t + cx，物理加速度约为 2a。
                ax = float(coeff_x[0]) if len(coeff_x) == 3 else 0.0
                ay = float(coeff_y[0]) if len(coeff_y) == 3 else 0.0
                accel = float(np.hypot(2.0 * ax, 2.0 * ay))
            except Exception:
                accel = 0.0
            if max_curve > self._draw_kinematic_max_curve_px or accel > self._draw_kinematic_max_accel_px_per_ms2:
                deg = 1
                coeff_x, _ = self._polyfit_predict_1d(t_ms[mask], x[mask], deg)
                coeff_y, _ = self._polyfit_predict_1d(t_ms[mask], y[mask], deg)

        # 按轨迹长度自适应采样，避免画出来一段一段折线。
        sample_count = int(max(8, min(self._draw_kinematic_max_samples,
                                      max(8.0, chord_len) / self._draw_kinematic_sample_step_px)))
        sample_t = np.linspace(float(t_ms[0]), float(t_ms[-1]), sample_count)
        px = np.polyval(coeff_x, sample_t)
        py = np.polyval(coeff_y, sample_t)
        sample_ts = np.linspace(float(ts[0]), float(ts[-1]), sample_count)
        return [(float(xx), float(yy), int(round(tt))) for xx, yy, tt in zip(px, py, sample_ts)]

    @staticmethod
    def _theme_bgr(seed: int) -> Tuple[int, int, int]:
        """赛博风头部主色。以紫粉系为主，尾部会自动过渡到冷蓝。"""
        palettes = [
            (255, 80, 235),   # magenta / pink, BGR
            (255, 105, 245),
            (255, 150, 220),
        ]
        return palettes[int(seed) % len(palettes)]

    @staticmethod
    def _scale_color(color: Tuple[int, int, int], s: float) -> Tuple[int, int, int]:
        ss = max(0.0, min(2.0, float(s)))
        return tuple(int(max(0, min(255, round(float(c) * ss)))) for c in color)

    @staticmethod
    def _blend_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        tt = max(0.0, min(1.0, float(t)))
        return tuple(int(max(0, min(255, round(a[i] * (1.0 - tt) + b[i] * tt)))) for i in range(3))

    def _draw_cyber_trail_roi(
        self,
        img: np.ndarray,
        pts: List[Tuple[int, int, int]],
        head_color: Tuple[int, int, int],
        show_head: bool = True,
    ) -> None:
        """v7.3：ROI 2x 超采样的颜色增强赛博光轨。

        设计目标：
          - 在 v7.2 基础上只增强颜色饱和度和亮度，线宽结构保持稳定；
          - 颜色从冷蓝 -> 蓝紫 -> 紫粉 -> 白尖端；
          - 只在轨迹 ROI 内做 2x 绘制和小范围 glow，不做全屏特效；
          - 不做粒子，保持实时性。
        """
        if len(pts) < 2:
            return

        # 轻量平滑：减少点状折线感，但不改变检测/命中逻辑。
        smoothed: List[Tuple[int, int, int]] = [pts[0]]
        for i in range(1, len(pts) - 1):
            x = int(round((pts[i - 1][0] + 2.0 * pts[i][0] + pts[i + 1][0]) / 4.0))
            y = int(round((pts[i - 1][1] + 2.0 * pts[i][1] + pts[i + 1][1]) / 4.0))
            smoothed.append((x, y, pts[i][2]))
        smoothed.append(pts[-1])
        pts = smoothed

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = 18
        x0 = max(0, int(min(xs) - margin))
        y0 = max(0, int(min(ys) - margin))
        x1 = min(int(img.shape[1]), int(max(xs) + margin + 1))
        y1 = min(int(img.shape[0]), int(max(ys) + margin + 1))
        if x1 <= x0 or y1 <= y0:
            return

        roi_w = int(x1 - x0)
        roi_h = int(y1 - y0)
        if roi_w <= 2 or roi_h <= 2:
            return

        # 小 ROI 用 2x 超采样；ROI 太大时自动退回 1x，避免极端长轨迹拖慢。
        roi_area = roi_w * roi_h
        scale = 2 if roi_area <= 120_000 else 1
        sw, sh = roi_w * scale, roi_h * scale
        glow = np.zeros((sh, sw, 3), dtype=np.uint8)
        core = np.zeros((sh, sw, 3), dtype=np.uint8)

        local_pts = [
            (int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale)), p[2])
            for p in pts
        ]
        n = len(local_pts)

        # 多段高级渐变，不再单调：冷蓝 -> 蓝紫 -> 紫粉 -> 近白。
        # v7.3 color boost：提高饱和度，减少“发白发淡”的感觉。
        c_tail = (255, 230, 0)     # saturated cyan-electric blue, BGR
        c_mid0 = (255, 95, 40)     # vivid cold blue bridge
        c_mid  = (255, 30, 195)    # saturated blue-violet
        c_head = (245, 36, 255)    # saturated magenta/pink
        c_tip  = (255, 235, 255)   # bright tip, but not pure washed-out white

        def ramp(t: float) -> Tuple[int, int, int]:
            t = max(0.0, min(1.0, float(t)))
            if t < 0.28:
                return self._blend_color(c_tail, c_mid0, t / 0.28)
            if t < 0.68:
                return self._blend_color(c_mid0, c_mid, (t - 0.28) / 0.40)
            if t < 0.90:
                return self._blend_color(c_mid, c_head, (t - 0.68) / 0.22)
            return self._blend_color(c_head, c_tip, (t - 0.90) / 0.10)

        for i in range(1, n):
            p0 = local_pts[i - 1]
            p1 = local_pts[i]
            dist = math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])) / max(1, scale)
            if dist < 0.35:
                continue

            t = float(i) / float(max(1, n - 1))
            base = ramp(pow(t, 1.06))
            # 发光强度头部更强，尾巴更轻；但比 v7.1 稍微饱满一点。
            energy = 0.33 + 0.67 * pow(t, 1.28)
            # 色彩增强：主线减少白化，亮芯保留白亮，整体更有颜色。
            glow_col = self._scale_color(base, 0.46 + 0.36 * energy)
            mid_col = self._blend_color(base, (255, 255, 255), 0.08 + 0.10 * energy)
            core_col = self._blend_color(base, (255, 255, 255), 0.58 + 0.20 * energy)
            hot_col = self._blend_color(base, (255, 255, 255), 0.86)

            # v7.2：在保持精致的前提下稍微加粗，并让后段更有存在感。
            # 实际宽度约：glow 2.4~3.8px, mid 1.5~2.4px, core 1~1.5px。
            glow_th = max(2, int(round((2.35 + 1.45 * energy) * scale)))
            mid_th = max(1, int(round((1.20 + 0.72 * energy) * scale)))
            core_th = max(1, int(round((0.78 + 0.12 * energy) * scale)))

            cv2.line(glow, (p0[0], p0[1]), (p1[0], p1[1]), glow_col, glow_th, cv2.LINE_AA)
            cv2.line(core, (p0[0], p0[1]), (p1[0], p1[1]), mid_col, mid_th, cv2.LINE_AA)
            cv2.line(core, (p0[0], p0[1]), (p1[0], p1[1]), core_col, core_th, cv2.LINE_AA)
            # 后半段追加更精致的针状亮芯，并在尾后段开始轻轻出现，增强层次。
            if t > 0.30:
                accent_th = max(1, int(round((0.44 + 0.08 * energy) * scale)))
                cv2.line(core, (p0[0], p0[1]), (p1[0], p1[1]), hot_col, accent_th, cv2.LINE_AA)

        # 精致小弹头：不做大光球，只做白芯 + 粉紫包边 + 极短前向尖刺。
        if show_head:
            hx, hy = int(local_pts[-1][0]), int(local_pts[-1][1])
            head_base = ramp(1.0)
            halo = self._blend_color(head_base, (255, 255, 255), 0.18)
            hot = self._blend_color(head_base, (255, 255, 255), 0.82)
            cv2.circle(glow, (hx, hy), max(2, int(round(3.1 * scale))), self._scale_color(halo, 0.68), -1, cv2.LINE_AA)
            cv2.circle(core, (hx, hy), max(1, int(round(2.15 * scale))), hot, -1, cv2.LINE_AA)
            cv2.circle(core, (hx, hy), max(1, int(round(0.72 * scale))), (255, 255, 255), -1, cv2.LINE_AA)
            if len(local_pts) >= 2:
                dx = float(local_pts[-1][0] - local_pts[-2][0])
                dy = float(local_pts[-1][1] - local_pts[-2][1])
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    ux, uy = dx / norm, dy / norm
                    tip_len = 3.4 * scale
                    tipx = int(round(hx + ux * tip_len))
                    tipy = int(round(hy + uy * tip_len))
                    cv2.line(core, (hx, hy), (tipx, tipy), (255, 255, 255), max(1, int(round(0.55 * scale))), cv2.LINE_AA)

        # 只在小画布上做很轻的 glow，再缩回原 ROI，边缘更细腻。
        sigma = 1.22 * scale if scale > 1 else 1.10
        glow_blur = cv2.GaussianBlur(glow, (0, 0), sigma)
        if scale > 1:
            glow_small = cv2.resize(glow_blur, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
            core_small = cv2.resize(core, (roi_w, roi_h), interpolation=cv2.INTER_AREA)
        else:
            glow_small = glow_blur
            core_small = core

        roi = img[y0:y1, x0:x1]
        # v7.2：略微提高光轨存在感，但仍然靠局部 ROI 控制开销。
        cv2.addWeighted(roi, 1.0, glow_small, 0.98, 0.0, dst=roi)
        cv2.addWeighted(roi, 1.0, core_small, 1.34, 0.0, dst=roi)

    def draw(
        self, img, accepted_clusters: Iterable[dict],
        box_color=(0, 255, 255), text_color=(0, 255, 255),
        draw_debug_boxes: bool = False,
        show_head: bool = True,
    ) -> None:
        """Strict display-only renderer: confirmed bullet boxes only.

        Python fallback version. It intentionally does not draw candidate tracks,
        hold/miss residual tracks, raw/stable/probation candidates, or historical
        short trails. Only clusters with display_bullet_id >= 0 are shown.
        """
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
            display_bullet_id = _to_int(c.get('display_bullet_id', -1), -1)
            if display_bullet_id < 0:
                continue
            cx, cy = _cluster_center(c)
            bx1, by1, bx2, by2 = _bbox_from_cluster(c, cx, cy)
            phase_str = str(c.get('phase', ''))
            label = f"b_{display_bullet_id}"
            if phase_str == 'maintain_turn':
                label += '*'
            elif phase_str.startswith('ghost_capture'):
                label += '~'
            confirmed.append((display_bullet_id, bx1, by1, bx2, by2, label))

        if not confirmed:
            return

        for display_bullet_id, bx1, by1, bx2, by2, label in confirmed:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), BLACK, BBOX_THICKNESS + 2, cv2.LINE_AA)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), YELLOW, BBOX_THICKNESS, cv2.LINE_AA)
            tx = min(max(5, bx1), max(5, img.shape[1] - 90))
            ty = max(18, by1 - 6)
            if ty <= 18 and by2 + 18 < img.shape[0]:
                ty = by2 + 18
            ty = min(max(18, ty), max(18, img.shape[0] - 6))
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        ID_FONT_SCALE, BLACK, ID_THICKNESS + 3, cv2.LINE_AA)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        ID_FONT_SCALE, YELLOW, ID_THICKNESS, cv2.LINE_AA)
