

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 事件类型常量
# ---------------------------------------------------------------------------
EVENT_TURN      = "turn"       # 子弹轨迹发生方向转折
EVENT_TERMINATE = "terminate"  # 子弹轨迹终止（被 kill 或消失）
EVENT_POINT     = "point"      # 实时轨迹点采样：用于 UE5 实时可视化，也可驱动 4ms 级 segment 命中判定

# 终止原因（来自 LinearMotionFilter 内部，直接透传）
REASON_OFFSET_EXCEED   = "offset_exceed"
REASON_RECENT_OFFSET   = "recent_offset_exceed"
REASON_BBOX_UNSTABLE   = "bbox_unstable"
REASON_BBOX_EMA_DRIFT  = "bbox_ema_drift"
REASON_OVERTURN        = "overturn"
REASON_STATIC          = "static"
REASON_MISSING         = "missing"
REASON_PROBATION       = "probation"
REASON_PROBATION_OFFSET = "probation_offset"
REASON_UNKNOWN         = "unknown"
REASON_VISUAL_POINT    = "visual_point"  # 实时点流原因：每个新轨迹点直接推送给 UE5

# 命中匹配方法
MATCH_SEGMENT_INTERSECT = "segment_intersect"  # 线段穿入扩张 bbox
MATCH_POINT_IN_BBOX     = "point_in_bbox"      # 事件点落入扩张 bbox
MATCH_POINT_NEAR_BBOX   = "point_near_bbox"    # 事件点在扩张 bbox 边缘附近
MATCH_POINT_ONLY        = "point_only"         # approach_valid=False，退化为点判定


# ---------------------------------------------------------------------------
# BulletEvent — 客户端子弹事件
# ---------------------------------------------------------------------------

@dataclass
class BulletEvent:
    """
    客户端子弹过滤器产生的稀疏事件，通过 UDP 发往服务器。

    发送策略：
      turn_event     : 每次 maintain_turn 时发一次，不重发
      terminate_event: 发出后延迟 10~20ms 重发一次，服务端按唯一键幂等去重

    坐标空间：
      event_x/y 和 approach_x/y 均为事件相机原始分辨率下的像素坐标。
      服务器端判定时需根据 orig_w/orig_h 换算到 frame_w/frame_h 空间。
    """

    # ---- 路由字段 ----
    # ---- 路由字段（camera_sn 为全链路唯一主键）----
    camera_sn:    str          # 相机唯一硬件序列号（如 "4110046947"），全链路主键
    camera_alias: str  = ""    # 可选显示名，不参与逻辑，默认空字符串
    bullet_id:   int   = 0         # 子弹 id，来自 LinearMotionFilter 内部
    stable_track_id: int = -1      # Event worker stable trajectory id; preferred for cross-stage track evidence
    display_bullet_id: int = -1    # Display/preview bullet id when available
    global_bullet_id: str = ""     # Optional globally unique bullet id emitted by newer event workers

    # ---- 去重字段 ----
    seq_id:       int  = 0    # 同一 (camera_sn, bullet_id) 内的递增序号，从 0 开始
    is_retransmit: bool = False  # 是否为重发包（terminate_event 重发时置 True）

    # ---- 事件核心字段 ----
    event_type:   str  = EVENT_TERMINATE  # "turn" 或 "terminate"
    timestamp_us: int  = 0               # 事件相机硬件时间戳（微秒）

    # ---- 坐标字段 ----
    event_x:     float = 0.0   # 事件点 cx（最后一个有效检测点）
    event_y:     float = 0.0
    approach_x:  float = 0.0   # 接近段起始点（反向遍历到的有效前序点）
    approach_y:  float = 0.0
    approach_len_px: float = 0.0  # approach_point 到 event_point 的距离（像素）
    approach_valid:  bool  = False  # False 时命中模块退化为纯点判定

    # ---- 坐标空间 ----
    orig_w: int = 0   # 子弹坐标所在的图像宽度（事件相机原始分辨率）
    orig_h: int = 0   # 子弹坐标所在的图像高度

    # ---- 运动特征 ----
    avg_speed_px_per_ms: float = 0.0  # 最近几帧平均速度（像素/毫秒）
    segment_index:       int   = 0    # 当前轨迹段索引（每次 maintain_turn 后 +1）

    # ---- turn_event 专用字段 ----
    turn_angle_deg: float = 0.0  # 转折角度（度），非 turn 事件时为 0

    # ---- terminate_event 专用字段 ----
    terminate_reason: str = REASON_UNKNOWN  # 终止原因，来自 LinearMotionFilter

    # ---- 可选置信度 ----
    confidence: float = 1.0  # 0.0~1.0，seq_id 跳号时由服务端降低

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

    # ---- 连续轨迹质量字段：用于过滤人身上冒出来的伪 bullet ----
    track_point_count: int = 0
    track_path_len_px: float = 0.0
    track_duration_ms: float = 0.0
    track_displacement_px: float = 0.0
    track_straightness: float = 0.0
    track_line_error_px: float = 999.0

    # Optional C++ shot-link context.  These fields are produced by the C++
    # physical-shot bounce linker and are debug/decision context only: they do
    # not by themselves prove a human HIT.
    shot_link_type: str = ""
    shot_link_reason: str = ""
    shot_link_parent_bullet_id: int = -1
    shot_link_parent_segment_index: int = -1
    shot_link_gap_ms: float = 0.0
    shot_link_gap_dist_px: float = 0.0
    shot_link_angle_deg: float = 0.0

    # Debug-only context from the event-camera human-mask prefilter.
    # This is not used for motion tracking itself, but lets hit_judge/debug logs
    # explain whether the upstream hard mask removed events near this bullet.
    event_human_mask_stats: Dict[str, object] = field(default_factory=dict)

    def unique_key(self) -> tuple:
        """服务端幂等去重键：(camera_sn, bullet_id, seq_id)"""
        return (self.camera_sn, self.bullet_id, self.seq_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BulletEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "BulletEvent":
        return cls.from_dict(json.loads(data.decode("utf-8")))


# ---------------------------------------------------------------------------
# PersonSnapshot — 单个人物框快照（PersonHistoryEntry 的子结构）
# ---------------------------------------------------------------------------

@dataclass
class PersonSnapshot:
    """
    服务器端某一帧中一个已确认人物的检测结果快照。
    poly 和 mask 为可选字段，第一版判定只用 bbox，后续增强时再引入。

    stable_id 语义：
      该字段是 CameraInstanceStabilizer 分配的局部 track id。
      不等于全局选手编号，不保证跨遮挡/重现后一致。
    """
    stable_id:  int
    bbox:       Tuple[int, int, int, int]   # (x, y, w, h)，像素坐标
    centroid:   Tuple[float, float]          # (cx, cy)
    confirmed:  bool = True
    held:       bool = False                 # True 表示这是 hold 框（位置已过时）
    # 以下字段第一版不参与判定，仅留存以备后续使用
    poly:       Optional[List[List[int]]] = None
    mask_area:  float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Tuple 序列化为 list，反序列化时还原
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PersonSnapshot":
        d2 = dict(d)
        if d2.get("bbox") is not None:
            d2["bbox"] = tuple(d2["bbox"])
        if d2.get("centroid") is not None:
            d2["centroid"] = tuple(d2["centroid"])
        return cls(**{k: v for k, v in d2.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# PersonHistoryEntry — 服务器端每帧人物历史快照
# ---------------------------------------------------------------------------

@dataclass
class PersonHistoryEntry:
    """
    服务器端主循环每次完成 stabilizer.update() 后推入历史缓冲的一条记录。

    入缓冲规则（第一版）：
      只存 confirmed=True, held=False 的人物框。
      held=True 的框因为位置已过时，第一版不参与命中判定。

    时间精度说明：
      timestamp_us 来自 TSF1 包头的硬件时间戳，和子弹事件的 timestamp_us 是同一时钟。
      但人物历史的更新粒度由 TSF1 帧率决定（约 40ms/帧），
      而子弹事件的时间精度由客户端更新粒度决定（约 3ms/帧）。
      两者不对称，判定时必须显式建模时间误差 dt_ms。
    """
    camera_sn:    str          # 相机唯一硬件序列号
    camera_alias: str  = ""    # 可选显示名
    timestamp_us: int   = 0
    frame_id:     int   = 0
    frame_w:      int   = 0  # 该帧图像宽度（人物框坐标所在分辨率）
    frame_h:      int   = 0  # 该帧图像高度
    persons:      List[PersonSnapshot] = field(default_factory=list)
    # 记录时的墙钟时间，供调试用，不参与判定
    wall_time:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["persons"] = [p.to_dict() for p in self.persons]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PersonHistoryEntry":
        d2 = dict(d)
        d2["persons"] = [PersonSnapshot.from_dict(p) for p in d2.get("persons", [])]
        return cls(**{k: v for k, v in d2.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ScoreBreakdown — 命中评分分量
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """命中评分各分量，供调试和日志使用。"""
    time_score:   float = 0.0   # 时间对齐分：dt_ms 越小越高
    space_score:  float = 0.0   # 空间分：线段穿入 > 点在 bbox 内
    event_score:  float = 0.0   # 事件类型分：turn > terminate
    speed_score:  float = 0.0   # 速度分：avg_speed_px_per_ms 越高越可信
    loss_penalty: float = 0.0   # 丢包惩罚：seq_id 跳号时为负值

    @property
    def total(self) -> float:
        return (self.time_score + self.space_score +
                self.event_score + self.speed_score + self.loss_penalty)

    def to_dict(self) -> dict:
        return {
            "time_score":   round(self.time_score,   3),
            "space_score":  round(self.space_score,  3),
            "event_score":  round(self.event_score,  3),
            "speed_score":  round(self.speed_score,  3),
            "loss_penalty": round(self.loss_penalty, 3),
            "total":        round(self.total,        3),
        }


# ---------------------------------------------------------------------------
# HitCandidate — 命中判定模块输出的候选结果
# ---------------------------------------------------------------------------

@dataclass
class HitCandidate:
    """
    HitJudge 对每个 BulletEvent 输出的命中候选结果。

    设计原则：
      HitJudge 不做硬判决（命中/未命中），只输出带分数的候选。
      是否宣告命中，由消费端根据 total_score 阈值决定。
      这样调整阈值时不需要改 HitJudge 代码。

    stable_id 语义：
      该字段是 CameraInstanceStabilizer 的局部 track id。
      不等于全局选手编号。消费端若需累计同一物理人的命中次数，
      必须另建 player identity 映射层。
    """

    # ---- 溯源字段 ----
    camera_sn:    str          # 相机唯一硬件序列号
    camera_alias: str  = ""    # 可选显示名
    bullet_id:    int  = 0
    seq_id:       int  = 0  # 对应的 BulletEvent.seq_id，方便关联日志
    stable_id:    int  = 0  # 被命中的人物 track id（局部，非全局）
    global_player_id: Optional[str] = None  # 由外部玩家身份融合模块后处理填入，命中模块不生成

    # ---- 时间字段 ----
    event_ts:     int   = 0    # 子弹事件的 timestamp_us
    person_ts:    int   = 0    # 匹配到的人物历史快照的 timestamp_us
    dt_ms:        float = 0.0  # abs(person_ts - event_ts) / 1000，越小越可信
    judge_wall_time: float = field(default_factory=time.time)  # 判定产生时的墙钟

    # ---- 事件字段 ----
    event_type:      str   = EVENT_TERMINATE
    terminate_reason: str  = REASON_UNKNOWN
    turn_angle_deg:  float = 0.0

    # ---- 几何字段 ----
    event_point:     Tuple[float, float] = (0.0, 0.0)  # 换算到 frame 坐标系后的事件点
    approach_point:  Tuple[float, float] = (0.0, 0.0)  # 换算后的接近段起始点
    approach_valid:  bool  = False
    match_method:    str   = MATCH_POINT_ONLY

    # ---- 连续轨迹质量字段，透传到 hit_candidate JSONL ----
    track_point_count: int = 0
    track_path_len_px: float = 0.0
    track_duration_ms: float = 0.0
    track_displacement_px: float = 0.0
    track_straightness: float = 0.0
    track_line_error_px: float = 999.0

    # ---- 人物框字段 ----
    person_bbox:     Tuple[int, int, int, int] = (0, 0, 0, 0)
    person_centroid: Tuple[float, float]       = (0.0, 0.0)
    person_confirmed: bool = True
    person_held:     bool  = False

    # ---- 评分字段 ----
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    @property
    def total_score(self) -> float:
        return self.score_breakdown.total

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score_breakdown"] = self.score_breakdown.to_dict()
        d["total_score"]     = round(self.total_score, 3)
        # 明确标注 stable_id 语义
        d["stable_id_is_global"] = False
        d["stable_id_semantics"] = "per-camera temporary track id, not global player id"
        return d

    def to_json_line(self) -> str:
        """序列化为单行 JSON，用于 JSONL 日志追加写入。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
