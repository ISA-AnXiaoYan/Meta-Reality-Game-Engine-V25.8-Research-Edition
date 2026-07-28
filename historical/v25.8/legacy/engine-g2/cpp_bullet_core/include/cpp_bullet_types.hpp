#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bullet_cpp {

struct Point3 {
    double x = 0.0;
    double y = 0.0;
    int64_t ts = -1;
    bool operator==(const Point3& o) const noexcept {
        return x == o.x && y == o.y && ts == o.ts;
    }
};

struct Point5 {
    double x = 0.0;
    double y = 0.0;
    double w = 1.0;
    double h = 1.0;
    int64_t ts = -1;
    bool operator==(const Point5& o) const noexcept {
        return x == o.x && y == o.y && w == o.w && h == o.h && ts == o.ts;
    }
};

template <class T>
class LimitedDeque {
public:
    explicit LimitedDeque(size_t maxlen = 0) : maxlen_(maxlen) {}
    void set_maxlen(size_t n) { maxlen_ = n; trim(); }
    size_t maxlen() const { return maxlen_; }
    void clear() { data_.clear(); }
    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }
    const T& back() const { return data_.back(); }
    T& back() { return data_.back(); }
    const T& front() const { return data_.front(); }
    const T& operator[](size_t i) const { return data_[i]; }
    T& operator[](size_t i) { return data_[i]; }
    const std::deque<T>& raw() const { return data_; }
    std::deque<T>& raw() { return data_; }
    void push_back(const T& v) { data_.push_back(v); trim(); }
    void assign(const std::vector<T>& v) { data_.assign(v.begin(), v.end()); trim(); }
    std::vector<T> to_vector() const { return std::vector<T>(data_.begin(), data_.end()); }
    typename std::deque<T>::const_iterator begin() const { return data_.begin(); }
    typename std::deque<T>::const_iterator end() const { return data_.end(); }
private:
    void trim() {
        if (maxlen_ == 0) return;
        while (data_.size() > maxlen_) data_.pop_front();
    }
    size_t maxlen_ = 0;
    std::deque<T> data_;
};

struct TrackConfig {
    int history_len = 8;
    int bootstrap_frames = 1;
    double max_angle_deg = 16.0;
    double max_line_offset_px = 4.0;
    double min_step_px = 3.0;
    double min_total_displacement_px = 20.0;
    int max_missed_frames = 4;
    double same_direction_ratio_thresh = 0.75;
    int bullet_min_output_streak = 5;
    int min_valid_step_count = 2;
    double min_valid_step_ratio = 0.55;
    double maintain_max_offset_px = 5.0;
    int maintain_max_static_frames = 5;
    double maintain_max_bbox_std = 4.5;
    double probation_max_offset_px = 4.5;
    bool ghost_capture_enabled = true;
    double maintain_recent_offset_px = 4.0;
    int maintain_recent_offset_exceed_frames = 3;
    double maintain_bbox_ema_alpha = 0.25;
    double maintain_bbox_ema_drift_ratio = 0.45;
    int maintain_max_turn_count = 2;
    double maintain_max_cumulative_turn_deg = 90.0;
    double ghost_bounce_max_dist_px = 35.0;
    double ghost_bounce_min_speed_px_per_ms = 0.60;
    double same_track_bbox_reuse_min_disp_px = 6.0;
    double trigger_max_full_offset_ratio = 1.0;
    double trigger_max_sign_flip_ratio = 0.20;
    double trigger_min_avg_speed_px_per_ms = 0.60;
    int trigger_raw_min_valid_steps = 3;
    int trigger_sparse_burst_max_valid_steps = 2;
    double trigger_sparse_burst_min_disp_per_step_px = 18.0;
    double trigger_sparse_burst_min_total_disp_px = 35.0;
    double trigger_sparse_burst_max_path_over_net = 1.03;
    double trigger_old_raw_age_ms = 120.0;
    int trigger_old_raw_min_valid_steps = 4;
    double probation_new_min_extra_disp_px = 8.0;
    int probation_new_min_extra_valid_steps = 2;
    bool model_track_enabled = true;
    double model_soft_residual_px = 9.0;
    double model_hard_residual_px = 22.0;
    double model_outlier_residual_px = 32.0;
    double model_correction_gain = 0.35;
    double model_weak_correction_gain = 0.10;
    double model_min_init_speed_px_ms = 0.35;
    double model_max_accel_px_ms2 = 0.16;
    bool model_outlier_kill_enabled = true;
    int model_outlier_kill_frames = 1;
    bool raw_id_sticky_guard_enabled = true;
    double raw_id_sticky_guard_min_speed_px_ms = 0.35;
    double raw_id_sticky_guard_static_step_px = 2.2;
    double raw_id_sticky_guard_backward_px = 1.0;
    double raw_id_sticky_guard_residual_px = 24.0;

    // P16: confirmed bullet ID stitching.
    // When a real bullet crosses a grass/black-obstacle boundary, SpatterTracker may
    // leave the old raw_id on the obstacle and create a new raw_id for the same bullet.
    // This layer lets a newly confirmed segment inherit the old physical bullet_id /
    // display_bullet_id after geometric validation, instead of opening a new ID.
    bool bullet_id_stitch_enabled = true;
    double bullet_id_stitch_window_ms = 120.0;
    double bullet_id_stitch_corridor_px = 24.0;
    double bullet_id_stitch_min_dir_cos = 0.82;
    double bullet_id_stitch_min_speed_px_ms = 0.32;
    double bullet_id_stitch_max_size_ratio = 5.5;
    double bullet_id_stitch_predict_gain = 0.55;
    double bullet_id_stitch_active_max_dt_ms = 140.0;

    // P17: reverse-line stitching.
    // If the front part of a bullet is hidden by an obstacle, the new raw_id may
    // appear only after the gap. Forward prediction from the old sticky/raw_id point
    // may be unreliable, so the newly confirmed straight segment is projected
    // backwards to find the old physical bullet_id.
    bool bullet_id_stitch_reverse_enabled = true;
    double bullet_id_stitch_reverse_window_ms = 170.0;
    double bullet_id_stitch_reverse_corridor_px = 26.0;
    double bullet_id_stitch_reverse_min_dir_cos = 0.82;
    double bullet_id_stitch_reverse_min_local_disp_px = 10.0;
    double bullet_id_stitch_reverse_max_size_ratio = 5.8;

    // P19: physical-shot bounce link.  This deliberately stays in the C++
    // tracking layer, not hit_judge: it links two ballistic bullet_id segments
    // that are close in time/space but reverse direction after hitting a surface.
    // The link only restores a physical shot trajectory; it does not prove HIT.
    bool bullet_id_bounce_link_enabled = true;
    double bullet_id_bounce_link_window_ms = 260.0;
    double bullet_id_bounce_link_max_gap_px = 95.0;
    double bullet_id_bounce_link_adaptive_max_px = 190.0;
    double bullet_id_bounce_link_min_turn_angle_deg = 95.0;
    double bullet_id_bounce_link_min_speed_px_ms = 0.38;
    double bullet_id_bounce_link_max_size_ratio = 6.5;
    int bullet_id_bounce_link_candidate_max_points = 12;
};

struct AssociationConfig {
    double max_distance_px = 36.0;
    double direction_penalty_px = 10.0;
    double max_size_ratio = 3.0;
};

struct DrawConfig {
    bool draw_trajectory = true;
    int history_len = 40;
    int pre_confirm_history_len = 24;
    int hold_missed_frames = 2;
    double connect_max_gap_px = 72.0;
    bool kinematic_fit_enabled = true;
    bool kinematic_force_straight_line = true;
    double kinematic_residual_px = 0.0;
    double kinematic_max_curve_px = 0.0;
    double kinematic_max_accel_px_per_ms2 = 0.0;
    double kinematic_sample_step_px = 7.0;
    int draw_after_terminate_hold_ms = 0;
    bool extend_tail_after_terminate = false;
};

struct Cluster {
    int64_t id = -1;
    int64_t raw_id = -1;
    double x = 0.0;
    double y = 0.0;
    double width = 1.0;
    double height = 1.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct SliceStats {
    int n_points = 0;
    double total_disp = 0.0;
    double net_disp = 0.0;
    double max_offset = 0.0;
    double mean_offset = 0.0;
    double direction_ok_ratio = 0.0;
    double same_sign_ratio = 0.0;
    int valid_step_count = 0;
    double valid_step_ratio = 0.0;
    int total_step_count = 0;
    double path_over_net = 999.0;
    double sign_flip_ratio = 1.0;
};

struct DebugRow {
    std::map<std::string, double> numeric;
    std::map<std::string, int64_t> integer;
    std::map<std::string, std::string> text;
};

struct TrackState {
    int stable_track_id = -1;
    LimitedDeque<Point5> points;
    LimitedDeque<Point3> pre_confirm_points;
    LimitedDeque<Point3> display_points;
    int miss_count = 0;
    std::optional<int> bullet_id;
    bool bullet_active = false;
    std::optional<int> last_bullet_id;
    bool confirmed_once = false;
    std::optional<Cluster> last_cluster;
    int keep_streak = 0;
    int static_fail_count = 0;
    std::optional<int64_t> current_raw_id;
    LimitedDeque<int64_t> raw_id_history{4};
    int64_t first_seen_ts = -1;
    int64_t bullet_assign_ts = -1;
    int64_t probation_until_ts = -1;
    bool probation_passed = false;
    int probation_fail_count = 0;
    std::optional<std::pair<double,double>> probation_ref_pos;
    int probation_start_point_count = 0;
    int64_t probation_start_ts = -1;
    std::string birth_assign_kind;
    std::optional<int> display_bullet_id;
    std::optional<int> hold_display_bullet_id;
    LimitedDeque<Point3> hold_display_points;
    int64_t draw_hold_until_ts = -1;
    int64_t last_terminated_ts = -1;
    std::optional<std::pair<double,double>> last_terminated_pos;
    std::optional<std::pair<double,double>> last_terminated_dir;
    std::string last_terminated_reason;
    int segment_index = 0;
    std::optional<std::pair<double,double>> segment_dir;
    LimitedDeque<double> recent_widths{8};
    LimitedDeque<double> recent_heights{8};
    int recent_offset_exceed_count = 0;
    double bbox_ema_w = -1.0;
    double bbox_ema_h = -1.0;
    int turn_count = 0;
    double cumulative_turn_deg = 0.0;
    int64_t segment_turn_warmup_until_ts = -1;
    double bbox_reuse_min_disp_px = 0.0;
    LimitedDeque<Point3> obs_points{48};
    LimitedDeque<Point3> model_points{80};
    bool model_ready = false;
    double model_x = 0.0;
    double model_y = 0.0;
    double model_vx = 0.0;
    double model_vy = 0.0;
    int64_t model_last_ts = -1;
    double model_residual_px = -1.0;
    double model_pred_x = 0.0;
    double model_pred_y = 0.0;
    bool model_obs_used = false;
    bool model_outlier = false;
    int model_outlier_count = 0;
    std::string model_update_mode = "uninitialized";
    double model_speed_px_ms = 0.0;
    bool impact_candidate = false;
    double impact_x = 0.0;
    double impact_y = 0.0;
    std::string impact_state;
    int64_t impact_start_ts = -1;
    int impact_outlier_frames = 0;
    int64_t impact_last_ts = -1;
    int64_t segment_pool_last_push_ts = -1;
    int64_t confirmed_backfill_last_emit_ts = -1;
    LimitedDeque<Point3> confirmed_backfill_points{32};
    int post_outlier_rearm_count = 0;
    int64_t post_outlier_rearm_last_ts = -1;
    int post_outlier_rearm_last_window = 0;
    std::string p12_owner_mode;
    int p12_owner_bullet_id = -1;
    int p12_inherited_segment_index = -1;
    std::string p12_owner_reason;
    bool p12_owner_pending = false;
    std::string p12_owner_pending_mode;
    int64_t p12_owner_pending_since_ts = -1;
    bool p12_polluted_tail = false;

    // P19 shot/bounce-link debug fields. These are exported to bullet_track_debug
    // and BulletEventSender so a link can generate a turn-like event while still
    // keeping HIT decision conservative in hit_judge.
    std::string shot_link_type;
    std::string shot_link_reason;
    int shot_link_parent_bullet_id = -1;
    int shot_link_parent_segment_index = -1;
    int shot_link_parent_stable_track_id = -1;
    double shot_link_parent_x = 0.0;
    double shot_link_parent_y = 0.0;
    double shot_link_gap_ms = 0.0;
    double shot_link_gap_dist_px = 0.0;
    double shot_link_angle_deg = 0.0;
    double shot_link_score = 0.0;

    TrackState() = default;
    TrackState(int tid, int history_len, int pre_confirm_len, int draw_history_len)
        : stable_track_id(tid), points(history_len), pre_confirm_points(pre_confirm_len),
          display_points(draw_history_len), hold_display_points(draw_history_len) {}
};

struct RecentTerminated {
    int bullet_id = -1;
    int display_bullet_id = -1;
    int stable_track_id = -1;
    int64_t raw_id = -1;
    int64_t ts = -1;
    double x = 0.0;
    double y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double width = 1.0;
    double height = 1.0;
    int segment_index = 0;
    double speed_px_ms = 0.0;
    std::string reason;
};

struct RecentSegment {
    int bullet_id = -1;
    int display_bullet_id = -1;
    int stable_track_id = -1;
    int64_t raw_id = -1;
    int64_t start_ts = -1;
    int64_t end_ts = -1;
    double start_x = 0.0;
    double start_y = 0.0;
    double end_x = 0.0;
    double end_y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double speed_px_ms = 0.0;
    double width = 1.0;
    double height = 1.0;
    int segment_index = 0;
    std::string reason;
};

struct OcclusionOwner {
    int bullet_id = -1;
    int display_bullet_id = -1;
    int stable_track_id = -1;
    int64_t raw_id = -1;
    int64_t ts = -1;
    double x = 0.0;
    double y = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double speed_px_ms = 0.0;
    double width = 1.0;
    double height = 1.0;
    int segment_index = 0;
    std::string reason;
};

inline double hypot2(double x, double y) { return std::sqrt(x*x + y*y); }
inline double dot2(double ax, double ay, double bx, double by) { return ax*bx + ay*by; }

}  // namespace bullet_cpp
