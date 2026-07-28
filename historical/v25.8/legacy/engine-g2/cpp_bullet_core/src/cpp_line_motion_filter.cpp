#include "cpp_line_motion_filter.hpp"
#include <stdexcept>
#include <numeric>

namespace bullet_cpp {


static std::map<std::string,double> stats_to_map(const SliceStats& st) {
    return {
        {"n_points", (double)st.n_points}, {"total_disp", st.total_disp}, {"net_disp", st.net_disp},
        {"max_offset", st.max_offset}, {"mean_offset", st.mean_offset},
        {"direction_ok_ratio", st.direction_ok_ratio}, {"same_sign_ratio", st.same_sign_ratio},
        {"valid_step_count", (double)st.valid_step_count}, {"valid_step_ratio", st.valid_step_ratio},
        {"total_step_count", (double)st.total_step_count}, {"path_over_net", st.path_over_net},
        {"sign_flip_ratio", st.sign_flip_ratio}
    };
}

static double median_value(std::vector<double> v, double fallback = 0.0) {
    if (v.empty()) return fallback;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2) return v[n/2];
    return 0.5 * (v[n/2-1] + v[n/2]);
}

static double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}


static double mget(const std::map<std::string,double>& m, const std::string& k, double def=0.0) {
    auto it = m.find(k); return it == m.end() ? def : it->second;
}
static bool bget(const std::map<std::string,double>& m, const std::string& k, bool def=false) {
    auto it = m.find(k); return it == m.end() ? def : (it->second != 0.0);
}
static void put_stats(DebugRow& r, const SliceStats* full, const SliceStats* recent) {
    if (full) {
        r.numeric["total_disp_px"] = full->total_disp;
        r.numeric["max_offset_px"] = full->max_offset;
        r.numeric["mean_offset_px"] = full->mean_offset;
        r.numeric["direction_ok_ratio"] = full->direction_ok_ratio;
        r.numeric["same_sign_ratio"] = full->same_sign_ratio;
        r.integer["valid_step_count"] = full->valid_step_count;
        r.numeric["valid_step_ratio"] = full->valid_step_ratio;
        r.numeric["path_over_net"] = full->path_over_net;
        r.numeric["sign_flip_ratio"] = full->sign_flip_ratio;
    }
    if (recent) {
        r.numeric["recent_total_disp_px"] = recent->total_disp;
        r.numeric["recent_max_offset_px"] = recent->max_offset;
        r.numeric["recent_direction_ok_ratio"] = recent->direction_ok_ratio;
        r.numeric["recent_same_sign_ratio"] = recent->same_sign_ratio;
        r.integer["recent_valid_step_count"] = recent->valid_step_count;
        r.numeric["recent_valid_step_ratio"] = recent->valid_step_ratio;
        r.numeric["recent_path_over_net"] = recent->path_over_net;
        r.numeric["recent_sign_flip_ratio"] = recent->sign_flip_ratio;
    }
}
static int64_t raw_id_for_state(const TrackState& st) {
    return st.current_raw_id.value_or(-1);
}


static int64_t last_ts_point5(const std::vector<Point5>& pts) {
    return pts.empty() ? -1 : pts.back().ts;
}

static std::vector<Point5> last_n_point5(const LimitedDeque<Point5>& dq, int n) {
    auto v = dq.to_vector();
    if (n > 0 && (int)v.size() > n) v.erase(v.begin(), v.end() - n);
    return v;
}


CppLineMotionFilter::CppLineMotionFilter(const TrackConfig& track, const AssociationConfig& assoc, const DrawConfig& draw)
    : tc_(track), ac_(assoc), dc_(draw) {
    initialize_derived_parameters();
}

void CppLineMotionFilter::initialize_derived_parameters() {
    const auto& tc = tc_;
    const auto& ac = ac_;
    const auto& dc = dc_;
    draw_trajectory = dc.draw_trajectory;
    draw_connect_max_gap_px = dc.connect_max_gap_px;
    draw_hold_missed_frames = std::max(0, dc.hold_missed_frames);
    draw_history_len = std::max((int)dc.history_len, (int)tc.history_len);
    pre_confirm_history_len = std::max((int)dc.pre_confirm_history_len, (int)tc.history_len);
    history_len = (int)tc.history_len;
    bootstrap_frames = (int)tc.bootstrap_frames;
    max_angle_deg = tc.max_angle_deg;
    max_line_offset_px = tc.max_line_offset_px;
    min_step_px = tc.min_step_px;
    min_total_displacement_px = tc.min_total_displacement_px;
    max_missed_frames = tc.max_missed_frames;
    same_direction_ratio_thresh = tc.same_direction_ratio_thresh;
    bullet_min_output_streak = tc.bullet_min_output_streak;
    min_valid_step_count = tc.min_valid_step_count;
    min_valid_step_ratio = tc.min_valid_step_ratio;
    maintain_max_offset_px = tc.maintain_max_offset_px;
    maintain_max_static_frames = tc.maintain_max_static_frames;
    maintain_max_bbox_std = tc.maintain_max_bbox_std;
    probation_max_offset_px = tc.probation_max_offset_px;
    ghost_capture_enabled = tc.ghost_capture_enabled;
    maintain_recent_offset_px = tc.maintain_recent_offset_px;
    maintain_recent_offset_exceed_frames = tc.maintain_recent_offset_exceed_frames;
    maintain_bbox_ema_alpha = tc.maintain_bbox_ema_alpha;
    maintain_bbox_ema_drift_ratio = tc.maintain_bbox_ema_drift_ratio;
    maintain_max_turn_count = tc.maintain_max_turn_count;
    maintain_max_cumulative_turn_deg = tc.maintain_max_cumulative_turn_deg;
    ghost_bounce_max_dist_px = tc.ghost_bounce_max_dist_px;
    ghost_bounce_min_speed_px_per_ms = tc.ghost_bounce_min_speed_px_per_ms;
    same_track_bbox_reuse_min_disp_px = tc.same_track_bbox_reuse_min_disp_px;
    trigger_max_full_offset_ratio = tc.trigger_max_full_offset_ratio;
    trigger_max_sign_flip_ratio = tc.trigger_max_sign_flip_ratio;
    trigger_min_avg_speed_px_per_ms = tc.trigger_min_avg_speed_px_per_ms;
    trigger_raw_min_valid_steps = tc.trigger_raw_min_valid_steps;
    trigger_sparse_burst_max_valid_steps = tc.trigger_sparse_burst_max_valid_steps;
    trigger_sparse_burst_min_disp_per_step_px = tc.trigger_sparse_burst_min_disp_per_step_px;
    trigger_sparse_burst_min_total_disp_px = tc.trigger_sparse_burst_min_total_disp_px;
    trigger_sparse_burst_max_path_over_net = tc.trigger_sparse_burst_max_path_over_net;
    trigger_old_raw_age_ms = tc.trigger_old_raw_age_ms;
    trigger_old_raw_min_valid_steps = tc.trigger_old_raw_min_valid_steps;
    probation_new_min_extra_disp_px = tc.probation_new_min_extra_disp_px;
    probation_new_min_extra_valid_steps = tc.probation_new_min_extra_valid_steps;
    model_track_enabled = tc.model_track_enabled;
    model_soft_residual_px = tc.model_soft_residual_px;
    model_hard_residual_px = tc.model_hard_residual_px;
    model_outlier_residual_px = tc.model_outlier_residual_px;
    model_correction_gain = tc.model_correction_gain;
    model_weak_correction_gain = tc.model_weak_correction_gain;
    model_min_init_speed_px_ms = tc.model_min_init_speed_px_ms;
    model_max_accel_px_ms2 = tc.model_max_accel_px_ms2;
    model_outlier_kill_enabled = tc.model_outlier_kill_enabled;
    model_outlier_kill_frames = std::max(1, tc.model_outlier_kill_frames);
    raw_id_sticky_guard_enabled = tc.raw_id_sticky_guard_enabled;
    raw_id_sticky_guard_min_speed_px_ms = tc.raw_id_sticky_guard_min_speed_px_ms;
    raw_id_sticky_guard_static_step_px = tc.raw_id_sticky_guard_static_step_px;
    raw_id_sticky_guard_backward_px = tc.raw_id_sticky_guard_backward_px;
    raw_id_sticky_guard_residual_px = tc.raw_id_sticky_guard_residual_px;

    bullet_id_stitch_enabled = tc.bullet_id_stitch_enabled;
    bullet_id_stitch_window_ms = std::max(20.0, tc.bullet_id_stitch_window_ms);
    bullet_id_stitch_corridor_px = std::max(8.0, tc.bullet_id_stitch_corridor_px);
    bullet_id_stitch_min_dir_cos = clampd(tc.bullet_id_stitch_min_dir_cos, 0.0, 0.99);
    bullet_id_stitch_min_speed_px_ms = std::max(0.05, tc.bullet_id_stitch_min_speed_px_ms);
    bullet_id_stitch_max_size_ratio = std::max(1.2, tc.bullet_id_stitch_max_size_ratio);
    bullet_id_stitch_predict_gain = clampd(tc.bullet_id_stitch_predict_gain, 0.0, 1.0);
    bullet_id_stitch_active_max_dt_ms = std::max(bullet_id_stitch_window_ms, tc.bullet_id_stitch_active_max_dt_ms);
    bullet_id_stitch_reverse_enabled = tc.bullet_id_stitch_reverse_enabled;
    bullet_id_stitch_reverse_window_ms = std::max(bullet_id_stitch_window_ms, tc.bullet_id_stitch_reverse_window_ms);
    bullet_id_stitch_reverse_corridor_px = std::max(10.0, tc.bullet_id_stitch_reverse_corridor_px);
    bullet_id_stitch_reverse_min_dir_cos = clampd(tc.bullet_id_stitch_reverse_min_dir_cos, 0.0, 0.99);
    bullet_id_stitch_reverse_min_local_disp_px = std::max(3.0, tc.bullet_id_stitch_reverse_min_local_disp_px);
    bullet_id_stitch_reverse_max_size_ratio = std::max(bullet_id_stitch_max_size_ratio, tc.bullet_id_stitch_reverse_max_size_ratio);

    bullet_id_bounce_link_enabled_ = tc.bullet_id_bounce_link_enabled;
    bullet_id_bounce_link_window_ms_ = std::max(40.0, tc.bullet_id_bounce_link_window_ms);
    bullet_id_bounce_link_max_gap_px_ = std::max(20.0, tc.bullet_id_bounce_link_max_gap_px);
    bullet_id_bounce_link_adaptive_max_px_ = std::max(bullet_id_bounce_link_max_gap_px_, tc.bullet_id_bounce_link_adaptive_max_px);
    bullet_id_bounce_link_min_turn_angle_deg_ = clampd(tc.bullet_id_bounce_link_min_turn_angle_deg, 45.0, 179.0);
    bullet_id_bounce_link_max_dir_cos_ = std::cos(bullet_id_bounce_link_min_turn_angle_deg_ * 3.14159265358979323846 / 180.0);
    bullet_id_bounce_link_min_speed_px_ms_ = std::max(0.05, tc.bullet_id_bounce_link_min_speed_px_ms);
    bullet_id_bounce_link_max_size_ratio_ = std::max(1.5, tc.bullet_id_bounce_link_max_size_ratio);
    bullet_id_bounce_link_candidate_max_points_ = std::max(3, tc.bullet_id_bounce_link_candidate_max_points);

    p12_forward_corridor_px_ = std::max(28.0, max_line_offset_px * 7.0);
    p12_forward_min_project_px_ = std::max(14.0, min_total_displacement_px * 0.55);
    p12_bounce_birth_px_ = std::max(64.0, ghost_bounce_max_dist_px * 1.8);
    p12_bounce_min_leave_px_ = std::max(18.0, min_total_displacement_px * 0.85);
    p12_bounce_min_speed_px_ms_ = std::max(0.55, ghost_bounce_min_speed_px_per_ms * 0.85);
    p12_residual_guard_px_ = std::max(110.0, ghost_bounce_max_dist_px * 3.0);
    segment_bounce_max_dist_px_ = std::max(70.0, ghost_bounce_max_dist_px * 2.0);
    segment_bounce_min_speed_px_ms_ = std::max(0.35, ghost_bounce_min_speed_px_per_ms * 0.45);
    segment_bounce_max_size_ratio_ = std::max(4.0, ac.max_size_ratio * 1.50);
    confirmed_backfill_min_speed_px_ms_ = std::max(0.45, trigger_min_avg_speed_px_per_ms * 0.75);
    post_outlier_rearm_reasons_ = {"model_outlier", "probation_reuse_motion", "probation_new_motion", "probation_offset", "probation", "recent_offset_exceed"};
    recent_window_points_ = std::min(std::max(4, history_len), 5);
    slow_window_points_ = std::min(std::max(recent_window_points_ + 2, 6), std::max(history_len, 10));
    trigger_min_recent_valid_steps_ = std::max(2, min_valid_step_count);
    trigger_min_recent_valid_ratio_ = std::max(0.65, min_valid_step_ratio);
    trigger_max_recent_offset_px_ = max_line_offset_px * 1.50;
    trigger_min_dir_ratio_ = std::max(0.72, same_direction_ratio_thresh * 0.90);
    trigger_min_same_sign_ = std::max(0.72, same_direction_ratio_thresh * 0.90);
    probation_continue_disp_px_ = std::max(8.0, min_total_displacement_px * 0.45);
    reuse_probation_birth_kinds_ = {"ballistic_rearm", "post_outlier_rearm", "same_track", "ghost", "ghost_capture", "ghost_capture_bounce", "segment_bounce", "segment_forward"};
    reuse_probation_min_disp_px_ = std::max(6.0, min_total_displacement_px * 0.30);
    reuse_probation_min_speed_px_ms_ = std::max(0.40, trigger_min_avg_speed_px_per_ms * 0.65);
    reuse_probation_max_offset_px_ = std::max(probation_max_offset_px * 1.35, max_line_offset_px * 1.80);
    terminate_miss_frames_ = std::max(3, max_missed_frames);
    recent_static_disp_px_ = std::max(0.8, min_step_px * 0.25);
    long_static_disp_px_ = std::max(2.0, min_step_px * 0.70);
    stationary_terminate_frames_ = maintain_max_static_frames;
    maintain_size_ratio_ = std::max(3.5, ac.max_size_ratio * 1.8);
    keep_tail_points_ = std::max(4, std::min(6, history_len));
    ghost_base_distance_px_ = std::max(200.0, ac.max_distance_px * 3.5);
    segment_turn_cos_ = std::cos(45.0 * M_PI / 180.0);
    draw_after_terminate_hold_us_ = std::max(0, dc.draw_after_terminate_hold_ms) * 1000LL;
    draw_extend_tail_after_terminate_ = dc.extend_tail_after_terminate;
    draw_spike_perp_px_ = std::max(12.0, max_line_offset_px * 3.0);
    draw_spike_min_leg_px_ = std::max(6.0, min_step_px * 2.0);
    draw_kinematic_fit_enabled_ = dc.kinematic_fit_enabled;
    draw_kinematic_force_straight_line_ = dc.kinematic_force_straight_line;
    draw_kinematic_base_residual_px_ = (dc.kinematic_residual_px > 0.0 ? dc.kinematic_residual_px : std::max(10.0, max_line_offset_px * 2.8));
    draw_kinematic_max_curve_px_ = dc.kinematic_max_curve_px;
    draw_kinematic_max_accel_px_per_ms2_ = dc.kinematic_max_accel_px_per_ms2;
    draw_kinematic_sample_step_px_ = std::max(3.0, dc.kinematic_sample_step_px);
}

Cluster CppLineMotionFilter::cluster_to_struct(int64_t raw_id, double x, double y, double w, double h) {
    Cluster c;
    c.id = raw_id;
    c.raw_id = raw_id;
    c.x = x;
    c.y = y;
    c.width = w;
    c.height = h;
    c.cx = x + 0.5 * w;
    c.cy = y + 0.5 * h;
    return c;
}

TrackState& CppLineMotionFilter::create_state() {
    int tid = next_track_id_++;
    auto it = tracks_.emplace(tid, TrackState(tid, history_len, pre_confirm_history_len, draw_history_len)).first;
    return it->second;
}

void CppLineMotionFilter::append_pre_confirm_point(TrackState& state, double x, double y, int64_t ts) {
    append_point_if_new(state.pre_confirm_points, Point3{(double)x, (double)y, (int64_t)ts});
}

void CppLineMotionFilter::append_display_point(TrackState& state, double x, double y, int64_t ts) {
    append_point_if_new(state.display_points, Point3{(double)x, (double)y, (int64_t)ts});
}

void CppLineMotionFilter::backfill_display_from_preconfirm(TrackState& state) {
    for (const auto& p : state.pre_confirm_points.raw()) append_display_point(state, p.x, p.y, p.ts);
}

std::vector<Point5> CppLineMotionFilter::trim_points(const LimitedDeque<Point5>& pts, int n) const {
    std::vector<Point5> v = pts.to_vector();
    if (n > 0 && (int)v.size() > n) v.erase(v.begin(), v.end() - n);
    return v;
}

std::pair<double,double> CppLineMotionFilter::predict_center(const TrackState& state, int64_t ts) const {
    if (state.points.size() == 0) return {0.0, 0.0};
    if (state.points.size() < 2) {
        const auto& p = state.points.back();
        return {p.x, p.y};
    }
    const auto& p1 = state.points[state.points.size() - 2];
    const auto& p2 = state.points[state.points.size() - 1];
    double dt = std::max(1.0, (double)(p2.ts - p1.ts));
    double vx = (p2.x - p1.x) / dt;
    double vy = (p2.y - p1.y) / dt;
    double fdt = std::max(0.0, (double)(ts - p2.ts));
    return {p2.x + vx * fdt, p2.y + vy * fdt};
}

std::optional<std::pair<double,double>> CppLineMotionFilter::track_direction(const TrackState& state) const {
    if (state.points.size() < 2) return std::nullopt;
    const auto& a = state.points.front();
    const auto& b = state.points.back();
    double dx = b.x - a.x, dy = b.y - a.y;
    double n = hypot2(dx, dy);
    if (n < 1e-6) return std::nullopt;
    return std::make_pair(dx / n, dy / n);
}

std::optional<SliceStats> CppLineMotionFilter::slice_stats(const std::vector<Point5>& pts) const {
    if (pts.size() < 2) return std::nullopt;
    const auto& first = pts.front();
    const auto& last = pts.back();
    double dx = last.x - first.x;
    double dy = last.y - first.y;
    double net = hypot2(dx, dy);
    double total = 0.0;
    int valid = 0;
    int flips = 0;
    int same_sign = 0;
    int total_steps = 0;
    std::vector<std::pair<double,double>> steps;
    for (size_t i = 1; i < pts.size(); ++i) {
        double sx = pts[i].x - pts[i-1].x;
        double sy = pts[i].y - pts[i-1].y;
        double sl = hypot2(sx, sy);
        total += sl;
        if (sl >= min_step_px) valid++;
        steps.push_back({sx, sy});
        total_steps++;
    }
    for (size_t i = 1; i < steps.size(); ++i) {
        double d = dot2(steps[i-1].first, steps[i-1].second, steps[i].first, steps[i].second);
        if (d >= 0) same_sign++; else flips++;
    }
    double mean_off = 0.0, max_off = 0.0;
    if (net > 1e-6) {
        for (const auto& p : pts) {
            double off = std::fabs((p.x - first.x) * dy - (p.y - first.y) * dx) / net;
            max_off = std::max(max_off, off);
            mean_off += off;
        }
        mean_off /= std::max<size_t>(1, pts.size());
    }
    double dir_ok = 0.0;
    if (net > 1e-6 && total_steps > 0) {
        double ux = dx / net, uy = dy / net;
        int ok = 0;
        for (const auto& s : steps) {
            double sl = hypot2(s.first, s.second);
            if (sl < 1e-6) continue;
            double cosv = dot2(s.first/sl, s.second/sl, ux, uy);
            if (cosv > std::cos(max_angle_deg * M_PI / 180.0)) ok++;
        }
        dir_ok = (double)ok / std::max(1, total_steps);
    }
    SliceStats st;
    st.n_points = (int)pts.size();
    st.total_disp = total;
    st.net_disp = net;
    st.max_offset = max_off;
    st.mean_offset = mean_off;
    st.direction_ok_ratio = dir_ok;
    st.same_sign_ratio = (steps.size() > 1 ? (double)same_sign / (double)(steps.size() - 1) : 1.0);
    st.valid_step_count = valid;
    st.valid_step_ratio = (double)valid / std::max(1, total_steps);
    st.total_step_count = total_steps;
    st.path_over_net = total / std::max(1e-6, net);
    st.sign_flip_ratio = (steps.size() > 1 ? (double)flips / (double)(steps.size() - 1) : 0.0);
    return st;
}

std::optional<SliceStats> CppLineMotionFilter::compute_stats(const LimitedDeque<Point5>& pts) const {
    return slice_stats(pts.to_vector());
}

double CppLineMotionFilter::estimate_speed_from_points(const std::vector<Point5>& pts) const {
    if (pts.size() < 2) return 0.0;
    double dist = hypot2(pts.back().x - pts.front().x, pts.back().y - pts.front().y);
    double dt_ms = std::max(0.001, (double)(pts.back().ts - pts.front().ts) / 1000.0);
    return dist / dt_ms;
}

double CppLineMotionFilter::estimate_recent_speed_px_per_ms(const TrackState& state) const {
    auto v = state.points.to_vector();
    if (v.size() > 4) v.erase(v.begin(), v.end() - 4);
    return estimate_speed_from_points(v);
}

int CppLineMotionFilter::promote_display_id(TrackState& state) {
    if (state.display_bullet_id.has_value()) return *state.display_bullet_id;
    int bid = state.bullet_id.value_or(next_bullet_id_++);
    state.bullet_id = bid;
    state.display_bullet_id = bid;
    displayed_bullet_ids_.insert(bid);
    return bid;
}

void CppLineMotionFilter::rebuild_raw_map() {
    std::unordered_map<int64_t, int> mapping;
    for (const auto& kv : tracks_) {
        int tid = kv.first;
        const auto& st = kv.second;
        if (st.current_raw_id.has_value()) mapping[*st.current_raw_id] = tid;
        for (auto raw_id : st.raw_id_history.raw()) mapping[raw_id] = tid;
    }
    raw_to_track_.swap(mapping);
}
void CppLineMotionFilter::purge_recent_terminated(int64_t ts) {
    recent_terminated_.erase(std::remove_if(recent_terminated_.begin(), recent_terminated_.end(), [&](const RecentTerminated& item){
        return ts - item.ts > ghost_reactivate_window_us_;
    }), recent_terminated_.end());
}
void CppLineMotionFilter::purge_recent_segments(int64_t ts) {
    recent_segments_.erase(std::remove_if(recent_segments_.begin(), recent_segments_.end(), [&](const RecentSegment& item){
        return ts - item.end_ts > recent_segment_pool_window_us_;
    }), recent_segments_.end());
}
void CppLineMotionFilter::purge_occlusion_owners(int64_t ts) {
    occlusion_owners_.erase(std::remove_if(occlusion_owners_.begin(), occlusion_owners_.end(), [&](const OcclusionOwner& item){
        return ts - item.ts > p12_owner_window_us_;
    }), occlusion_owners_.end());
}
bool CppLineMotionFilter::is_confirmed_owner_source(const TrackState& state) const {
    if (!state.bullet_id.has_value()) return false;
    int bid = *state.bullet_id;
    return state.display_bullet_id.has_value() || displayed_bullet_ids_.count(bid) > 0 || state.probation_passed;
}
void CppLineMotionFilter::push_occlusion_owner(const TrackState& state, int64_t ts, const std::string& reason) {
    if (!state.bullet_id.has_value() || !is_confirmed_owner_source(state)) return;
    if (ts < 0) return;
    int bid = *state.bullet_id;
    double px = 0.0, py = 0.0;
    if (!state.impact_state.empty() && (state.impact_x != 0.0 || state.impact_y != 0.0)) {
        px = state.impact_x; py = state.impact_y;
    } else if (state.model_ready) {
        px = state.model_x; py = state.model_y;
    } else if (state.last_cluster.has_value()) {
        px = state.last_cluster->cx; py = state.last_cluster->cy;
    } else return;
    double dx = 0.0, dy = 0.0;
    bool has_dir = false;
    if (state.segment_dir.has_value()) { dx = state.segment_dir->first; dy = state.segment_dir->second; has_dir = true; }
    if (!has_dir && state.last_terminated_dir.has_value()) { dx = state.last_terminated_dir->first; dy = state.last_terminated_dir->second; has_dir = true; }
    double dn = hypot2(dx, dy);
    if (has_dir && dn > 1e-6) { dx /= dn; dy /= dn; } else { dx = 0.0; dy = 0.0; }
    double w = state.last_cluster ? std::max(1.0, state.last_cluster->width) : 1.0;
    double h = state.last_cluster ? std::max(1.0, state.last_cluster->height) : 1.0;
    OcclusionOwner item;
    item.bullet_id = bid;
    item.display_bullet_id = state.display_bullet_id.value_or(-1);
    item.segment_index = state.segment_index;
    item.ts = ts;
    item.x = px; item.y = py;
    item.dir_x = dx; item.dir_y = dy;
    item.width = w; item.height = h;
    item.stable_track_id = state.stable_track_id;
    item.raw_id = state.current_raw_id.value_or(-1);
    item.reason = reason;
    item.speed_px_ms = state.model_ready ? state.model_speed_px_ms : estimate_recent_speed_px_per_ms(state);
    for (auto it = occlusion_owners_.rbegin(); it != occlusion_owners_.rend(); ++it) {
        if (it->bullet_id == bid && it->stable_track_id == state.stable_track_id && std::llabs(ts - it->ts) <= 6000) {
            *it = item;
            purge_occlusion_owners(ts);
            return;
        }
    }
    occlusion_owners_.push_back(item);
    purge_occlusion_owners(ts);
}
std::map<std::string,double> CppLineMotionFilter::candidate_motion_summary_for_owner(const TrackState& state) const {
    auto ok_stats = select_ballistic_rearm_suffix_stats(state, state.points.empty() ? -1 : state.points.back().ts);
    const bool ok = ok_stats.has_value();
    auto pts = state.points.to_vector();
    int local_points = (int)pts.size();
    double speed = ok ? estimate_recent_speed_px_per_ms(state) : estimate_recent_speed_px_per_ms(state);
    double total_disp = 0.0, max_offset = 999.0, dir_ratio = 0.0, same_sign = 0.0, path_over_net = 999.0, flip = 1.0;
    int valid_steps = 0;
    if (ok_stats.has_value()) {
        total_disp = ok_stats->total_disp;
        max_offset = ok_stats->max_offset;
        dir_ratio = ok_stats->direction_ok_ratio;
        same_sign = ok_stats->same_sign_ratio;
        path_over_net = ok_stats->path_over_net;
        flip = ok_stats->sign_flip_ratio;
        valid_steps = ok_stats->valid_step_count;
    }
    int backfill_n = (int)state.confirmed_backfill_points.size();
    return {
        {"ok", ok ? 1.0 : 0.0}, {"local_points", (double)local_points}, {"speed_px_ms", speed},
        {"total_disp", total_disp}, {"max_offset", max_offset}, {"direction_ok_ratio", dir_ratio},
        {"same_sign_ratio", same_sign}, {"path_over_net", path_over_net}, {"sign_flip_ratio", flip},
        {"valid_step_count", (double)valid_steps}, {"confirmed_backfill_n", (double)backfill_n}
    };
}
bool CppLineMotionFilter::is_near_occlusion_owner(const TrackState& state, int64_t ts) const {
    if (!state.last_cluster.has_value()) return false;
    const Cluster& c = *state.last_cluster;
    for (const auto& owner : occlusion_owners_) {
        int64_t dt = ts - owner.ts;
        if (dt < 0 || dt > p12_owner_guard_us_) continue;
        double dist = hypot2(c.cx - owner.x, c.cy - owner.y);
        double adaptive = p12_bounce_birth_px_ + std::max(0.0, owner.speed_px_ms) * ((double)dt / 1000.0) * 0.35;
        double allowed = std::min(p12_bounce_adaptive_px_, std::max(p12_bounce_birth_px_, adaptive));
        if (dist <= allowed) return true;
    }
    return false;
}
std::map<std::string,double> CppLineMotionFilter::try_occlusion_owner_capture_new_track(TrackState& state, int64_t ts) {
    std::map<std::string,double> out{{"ok", 0.0}, {"bullet_id", -1.0}, {"display_bullet_id", -1.0}, {"segment_index", -1.0}, {"mode", 0.0}};
    if (!state.last_cluster.has_value()) return out;
    const Cluster& c = *state.last_cluster;
    auto summary = candidate_motion_summary_for_owner(state);
    if (summary["ok"] < 0.5) return out;
    auto local_pts = state.points.to_vector();
    bool local_dir_ok = false;
    double local_dir_x = 0.0, local_dir_y = 0.0, local_disp = 0.0;
    if (local_pts.size() >= 2) {
        double ldx = local_pts.back().x - local_pts.front().x;
        double ldy = local_pts.back().y - local_pts.front().y;
        local_disp = hypot2(ldx, ldy);
        if (local_disp >= 5.0) { local_dir_x = ldx / local_disp; local_dir_y = ldy / local_disp; local_dir_ok = true; }
    }
    double best_score = -1e18;
    const OcclusionOwner* best = nullptr;
    int best_mode = 0; // 1 forward, 2 bounce
    for (const auto& owner : occlusion_owners_) {
        int64_t dt = ts - owner.ts;
        if (dt < 0 || dt > p12_owner_window_us_) continue;
        double vx = c.cx - owner.x, vy = c.cy - owner.y;
        double dist = hypot2(vx, vy);
        double proj = vx * owner.dir_x + vy * owner.dir_y;
        double perp = hypot2(vx - proj * owner.dir_x, vy - proj * owner.dir_y);
        bool have_owner_dir = (owner.dir_x != 0.0 || owner.dir_y != 0.0);
        double dir_cos = 1.0;
        if (have_owner_dir && local_dir_ok) dir_cos = local_dir_x * owner.dir_x + local_dir_y * owner.dir_y;
        if (have_owner_dir && local_dir_ok && dir_cos < 0.76) continue;
        bool forward_ok = have_owner_dir && proj >= p12_forward_min_project_px_ && perp <= p12_forward_corridor_px_;
        // P18: disable loose owner_bounce ID inheritance.  It was useful for recovery,
        // but in the real field it can merge a new shot B into old shot A after A exits.
        // Only same-direction forward owner capture is allowed now.
        bool bounce_ok = false;
        if (!forward_ok && !bounce_ok) continue;
        double score = 50.0 - perp * 0.85 - std::max(0.0, -proj) * 3.0 + summary["total_disp"] * 0.02 + summary["speed_px_ms"] + dir_cos * 8.0;
        if (score > best_score) { best_score = score; best = &owner; best_mode = forward_ok ? 1 : 2; }
    }
    if (!best) return out;
    state.bullet_id = best->bullet_id;
    state.display_bullet_id = best->display_bullet_id >= 0 ? std::optional<int>(best->display_bullet_id) : std::optional<int>(best->bullet_id);
    state.bullet_active = true;
    state.probation_passed = true;
    state.birth_assign_kind = best_mode == 1 ? "owner_forward" : "owner_bounce";
    state.segment_index = best->segment_index + 1;
    state.p12_owner_mode = state.birth_assign_kind;
    state.p12_owner_bullet_id = best->bullet_id;
    state.p12_inherited_segment_index = best->segment_index;
    state.p12_owner_reason = best->reason;
    if (state.display_bullet_id.has_value()) displayed_bullet_ids_.insert(*state.display_bullet_id);
    out["ok"] = 1.0; out["bullet_id"] = (double)best->bullet_id; out["display_bullet_id"] = (double)state.display_bullet_id.value_or(-1); out["segment_index"] = (double)state.segment_index; out["mode"] = (double)best_mode;
    return out;
}
void CppLineMotionFilter::push_recent_segment(TrackState& state, int64_t ts, const std::string& reason) {
    if (!state.bullet_id.has_value() || ts < 0) return;
    if (state.segment_pool_last_push_ts >= 0 && std::llabs(ts - state.segment_pool_last_push_ts) <= 3500) return;
    double px = 0.0, py = 0.0;
    if (!state.impact_state.empty() && (state.impact_x != 0.0 || state.impact_y != 0.0)) { px = state.impact_x; py = state.impact_y; }
    else if (state.model_ready) { px = state.model_x; py = state.model_y; }
    else if (state.last_cluster.has_value()) { px = state.last_cluster->cx; py = state.last_cluster->cy; }
    else return;
    double dx = 0.0, dy = 0.0;
    if (state.segment_dir.has_value()) { dx = state.segment_dir->first; dy = state.segment_dir->second; }
    else if (state.last_terminated_dir.has_value()) { dx = state.last_terminated_dir->first; dy = state.last_terminated_dir->second; }
    double dn = hypot2(dx, dy); if (dn > 1e-6) { dx /= dn; dy /= dn; }
    RecentSegment item;
    item.bullet_id = *state.bullet_id;
    item.display_bullet_id = state.display_bullet_id.value_or(-1);
    item.segment_index = state.segment_index;
    item.start_ts = state.points.empty() ? ts : state.points.front().ts;
    item.end_ts = ts;
    item.start_x = state.points.empty() ? px : state.points.front().x;
    item.start_y = state.points.empty() ? py : state.points.front().y;
    item.end_x = px; item.end_y = py;
    item.dir_x = dx; item.dir_y = dy;
    item.width = state.last_cluster ? std::max(1.0, state.last_cluster->width) : 1.0;
    item.height = state.last_cluster ? std::max(1.0, state.last_cluster->height) : 1.0;
    item.stable_track_id = state.stable_track_id;
    item.raw_id = state.current_raw_id.value_or(-1);
    item.reason = reason;
    item.speed_px_ms = state.model_ready ? state.model_speed_px_ms : estimate_recent_speed_px_per_ms(state);
    recent_segments_.push_back(item);
    state.segment_pool_last_push_ts = ts;
    purge_recent_segments(ts);
}
void CppLineMotionFilter::push_recent_terminated(const TrackState& state, int64_t ts, const std::string& reason) {
    if (!state.bullet_id.has_value() || !state.last_cluster.has_value()) return;
    RecentTerminated item;
    item.bullet_id = *state.bullet_id;
    item.display_bullet_id = state.display_bullet_id.value_or(-1);
    item.ts = ts >= 0 ? ts : state.last_cluster->raw_id;
    item.x = state.model_ready ? state.model_x : state.last_cluster->cx;
    item.y = state.model_ready ? state.model_y : state.last_cluster->cy;
    if (state.segment_dir.has_value()) { item.dir_x = state.segment_dir->first; item.dir_y = state.segment_dir->second; }
    else if (state.last_terminated_dir.has_value()) { item.dir_x = state.last_terminated_dir->first; item.dir_y = state.last_terminated_dir->second; }
    item.width = state.last_cluster->width;
    item.height = state.last_cluster->height;
    item.stable_track_id = state.stable_track_id;
    item.raw_id = state.current_raw_id.value_or(-1);
    item.reason = reason;
    item.segment_index = state.segment_index;
    item.speed_px_ms = state.model_ready ? state.model_speed_px_ms : estimate_recent_speed_px_per_ms(state);
    recent_terminated_.push_back(item);
}
void CppLineMotionFilter::deactivate_bullet(TrackState& state, int64_t ts, const std::string& reason) {
    if (state.bullet_active && state.bullet_id.has_value()) {
        state.last_bullet_id = *state.bullet_id;
        state.last_terminated_ts = state.last_cluster ? ts : -1;
        if (state.model_ready) state.last_terminated_pos = std::make_pair(state.model_x, state.model_y);
        else if (state.last_cluster) state.last_terminated_pos = std::make_pair(state.last_cluster->cx, state.last_cluster->cy);
        if (state.segment_dir.has_value()) state.last_terminated_dir = state.segment_dir;
        state.last_terminated_reason = reason;
        if (reason == "model_outlier" || reason == "recent_offset_exceed" || reason == "bbox_unstable" || reason == "bbox_ema_drift") {
            push_occlusion_owner(state, ts, reason);
        }
        push_recent_terminated(state, ts, reason);
        push_recent_segment(state, ts, reason);
    }
    state.bullet_active = false;
    state.bullet_id.reset();
    state.keep_streak = 0;
    state.static_fail_count = 0;
    state.probation_passed = false;
    state.probation_fail_count = 0;
    state.probation_until_ts = -1;
    state.probation_ref_pos.reset();
    state.probation_start_point_count = 0;
    state.probation_start_ts = -1;
    state.birth_assign_kind.clear();
    state.p12_owner_mode.clear();
    state.p12_owner_bullet_id = -1;
    state.p12_inherited_segment_index = -1;
    state.p12_owner_reason.clear();
    state.p12_owner_pending = false;
    state.p12_owner_pending_mode.clear();
    state.p12_owner_pending_since_ts = -1;
    state.p12_polluted_tail = false;
    state.shot_link_type.clear();
    state.shot_link_reason.clear();
    state.shot_link_parent_bullet_id = -1;
    state.shot_link_parent_segment_index = -1;
    state.shot_link_parent_stable_track_id = -1;
    state.shot_link_parent_x = 0.0;
    state.shot_link_parent_y = 0.0;
    state.shot_link_gap_ms = 0.0;
    state.shot_link_gap_dist_px = 0.0;
    state.shot_link_angle_deg = 0.0;
    state.shot_link_score = 0.0;
    if (draw_after_terminate_hold_us_ > 0 && state.display_bullet_id.has_value() && state.last_terminated_ts >= 0) {
        state.hold_display_bullet_id = *state.display_bullet_id;
        std::vector<Point3> src = state.model_points.size() >= 2 ? state.model_points.to_vector() : state.display_points.to_vector();
        state.hold_display_points.assign(src);
        state.draw_hold_until_ts = state.last_terminated_ts + draw_after_terminate_hold_us_;
    } else {
        state.hold_display_bullet_id.reset();
        state.hold_display_points.clear();
        state.draw_hold_until_ts = -1;
    }
    state.display_bullet_id.reset();
    state.recent_offset_exceed_count = 0;
    state.segment_turn_warmup_until_ts = -1;
    if (reason == "bbox_unstable" || reason == "bbox_ema_drift" || reason == "model_outlier") state.bbox_reuse_min_disp_px = same_track_bbox_reuse_min_disp_px;
    else state.bbox_reuse_min_disp_px = 0.0;
    // Keep points trim behavior: LimitedDeque already enforces maxlen.
}
std::optional<std::map<std::string,double>> CppLineMotionFilter::try_reuse_same_track_bullet(TrackState& state, int64_t ts) {
    if (!state.last_bullet_id.has_value() || state.last_terminated_ts < 0) return std::nullopt;
    if (ts - state.last_terminated_ts > same_track_reactivate_window_us_) return std::nullopt;
    if (state.last_terminated_reason == "model_outlier") return std::nullopt;
    if (state.last_terminated_reason == "bbox_unstable" || state.last_terminated_reason == "bbox_ema_drift") {
        if (!state.last_terminated_pos.has_value() || !state.last_cluster.has_value()) return std::nullopt;
        double disp = hypot2(state.last_cluster->cx - state.last_terminated_pos->first, state.last_cluster->cy - state.last_terminated_pos->second);
        double min_disp = std::max(same_track_bbox_reuse_min_disp_px, state.bbox_reuse_min_disp_px);
        if (disp < min_disp) return std::nullopt;
    }
    return std::map<std::string,double>{{"bullet_id", (double)*state.last_bullet_id}};
}
std::optional<std::map<std::string,double>> CppLineMotionFilter::try_reuse_recent_terminated_bullet(TrackState& state, int64_t ts) {
    if (!state.last_cluster.has_value()) return std::nullopt;
    const Cluster& c = *state.last_cluster;
    // P18: global ID-reuse direction guard.  Reusing a previous bullet_id is only
    // allowed when the new candidate segment moves in the same direction as the old
    // bullet.  This prevents bullet B from inheriting bullet A after A has left the view.
    auto local_pts = state.points.to_vector();
    bool local_dir_ok = false;
    double local_dir_x = 0.0, local_dir_y = 0.0, local_disp = 0.0;
    if (local_pts.size() >= 2) {
        double ldx = local_pts.back().x - local_pts.front().x;
        double ldy = local_pts.back().y - local_pts.front().y;
        local_disp = hypot2(ldx, ldy);
        if (local_disp >= 5.0) { local_dir_x = ldx / local_disp; local_dir_y = ldy / local_disp; local_dir_ok = true; }
    }
    double best_score = -1e18;
    const RecentTerminated* best = nullptr;
    for (const auto& item : recent_terminated_) {
        int64_t dt = ts - item.ts;
        if (dt < 0 || dt > ghost_reactivate_window_us_) continue;
        double dist = hypot2(c.cx - item.x, c.cy - item.y);
        double allowed = ghost_bounce_max_dist_px + std::max(0.0, item.speed_px_ms) * ((double)dt / 1000.0) * 0.35;
        allowed = std::max(ghost_bounce_max_dist_px, std::min(ghost_base_distance_px_, allowed));
        if (dist > allowed) continue;
        double size_ratio = std::max({c.width / std::max(1.0,item.width), item.width / std::max(1.0,c.width), c.height / std::max(1.0,item.height), item.height / std::max(1.0,c.height)});
        if (size_ratio > segment_bounce_max_size_ratio_) continue;
        double old_dn = hypot2(item.dir_x, item.dir_y);
        if (old_dn <= 1e-6) continue;
        double odx = item.dir_x / old_dn, ody = item.dir_y / old_dn;
        double vx = c.cx - item.x, vy = c.cy - item.y;
        double proj = vx * odx + vy * ody;
        double perp = hypot2(vx - proj * odx, vy - proj * ody);
        // No bounce-style reuse: the new point must be on/near the forward line.
        if (proj < -10.0) continue;
        double corridor = std::min(38.0, 18.0 + std::max(0.0, item.speed_px_ms) * ((double)dt / 1000.0) * 0.08);
        if (perp > corridor) continue;
        if (local_dir_ok) {
            double dir_cos = local_dir_x * odx + local_dir_y * ody;
            if (dir_cos < 0.76) continue;
        } else {
            // If the new segment has no direction yet, only allow a very tight forward reuse.
            if (perp > 14.0 || proj < 4.0) continue;
        }
        double score = -perp * 2.8 - std::max(0.0, -proj) * 6.0 - size_ratio * 8.0 + item.speed_px_ms * 6.0;
        if (score > best_score) { best_score = score; best = &item; }
    }
    if (!best) return std::nullopt;
    state.bullet_id = best->bullet_id;
    state.display_bullet_id = best->display_bullet_id >= 0 ? std::optional<int>(best->display_bullet_id) : std::optional<int>(best->bullet_id);
    state.birth_assign_kind = "recent_terminated";
    state.segment_index = best->segment_index + 1;
    state.bullet_active = true;
    if (state.display_bullet_id.has_value()) displayed_bullet_ids_.insert(*state.display_bullet_id);
    return std::map<std::string,double>{{"bullet_id", (double)best->bullet_id}, {"display_bullet_id", (double)state.display_bullet_id.value_or(-1)}, {"segment_index", (double)state.segment_index}};
}
std::optional<std::map<std::string,double>> CppLineMotionFilter::try_segment_pool_capture_new_track(TrackState& state, int64_t ts) {
    if (!state.last_cluster.has_value()) return std::nullopt;
    const int stitch_candidate_cap = std::max(segment_bounce_candidate_max_points_, bullet_id_bounce_link_candidate_max_points_);
    if ((int)state.points.size() > stitch_candidate_cap) return std::nullopt;

    const Cluster& c = *state.last_cluster;
    auto ok_stats = select_ballistic_rearm_suffix_stats(state, ts);
    bool suffix_ok = ok_stats.has_value();

    // P15: raw_id 粘在黑色障碍物上以后，真实子弹的新 raw_id 往往只有 1~3 个点时
    // 就会重新触发出生。如果这里仍强制要求完整 ballistic suffix，续接会来不及，
    // 同一颗子弹就会被分成多个 bullet_id。因此对 raw_id_sticky_handoff 来源开一个
    // “几何强约束快速继承”分支：必须沿旧弹道前方、在窄走廊内、且隐含速度合理。
    auto local_pts = state.points.to_vector();
    int local_points = (int)local_pts.size();
    double local_disp = 0.0;
    if (local_points >= 2) {
        const auto& pts = state.points.raw();
        local_disp = hypot2(pts.back().x - pts.front().x, pts.back().y - pts.front().y);
    }
    double local_speed = estimate_recent_speed_px_per_ms(state);
    bool local_dir_ok = false;
    double local_dir_x = 0.0, local_dir_y = 0.0;
    if (local_points >= 2 && local_disp >= 5.0) {
        const auto& pts = state.points.raw();
        local_dir_x = (pts.back().x - pts.front().x) / std::max(1e-6, local_disp);
        local_dir_y = (pts.back().y - pts.front().y) / std::max(1e-6, local_disp);
        local_dir_ok = true;
    }

    double best_score = -1e18;
    const RecentSegment* best = nullptr;
    std::string best_kind = "segment_pool";
    const int64_t shot_bounce_window_us = std::max<int64_t>(
        40000, (int64_t)std::llround(bullet_id_bounce_link_window_ms_ * 1000.0));

    for (const auto& seg : recent_segments_) {
        int64_t dt = ts - seg.end_ts;
        if (dt <= 0) continue;
        const bool segment_time_ok = (dt <= segment_bounce_dt_us_);
        const bool shot_bounce_time_ok = bullet_id_bounce_link_enabled_ && (dt <= shot_bounce_window_us);
        if (!segment_time_ok && !shot_bounce_time_ok) continue;

        const bool sticky_handoff = (seg.reason == "raw_id_sticky_handoff");
        double dt_ms = (double)dt / 1000.0;
        double dist = hypot2(c.cx - seg.end_x, c.cy - seg.end_y);
        double vx = c.cx - seg.end_x, vy = c.cy - seg.end_y;
        double implied_speed = dist / std::max(1e-6, dt_ms);

        double adaptive = 14.0 + std::max(seg.speed_px_ms, local_speed) * dt_ms * (sticky_handoff ? 0.95 : 0.85);
        double adaptive_cap = sticky_handoff ? std::max(segment_bounce_adaptive_max_px_, 170.0) : segment_bounce_adaptive_max_px_;
        double allowed = std::min(adaptive_cap, std::max(segment_bounce_max_dist_px_, adaptive));
        bool segment_gap_ok = segment_time_ok && (dist <= allowed);

        double shot_bounce_allowed = 18.0 + std::max(seg.speed_px_ms, local_speed) * dt_ms * 0.78;
        shot_bounce_allowed = std::min(
            bullet_id_bounce_link_adaptive_max_px_,
            std::max(bullet_id_bounce_link_max_gap_px_, shot_bounce_allowed));
        bool shot_bounce_gap_ok = shot_bounce_time_ok && (dist <= shot_bounce_allowed);
        if (!segment_gap_ok && !shot_bounce_gap_ok) continue;

        double size_ratio = std::max({c.width / std::max(1.0,seg.width), seg.width / std::max(1.0,c.width), c.height / std::max(1.0,seg.height), seg.height / std::max(1.0,c.height)});
        double allowed_sr = sticky_handoff ? std::max(segment_bounce_max_size_ratio_, 5.2) : segment_bounce_max_size_ratio_;
        bool segment_size_ok = segment_time_ok && (size_ratio <= allowed_sr);
        bool shot_bounce_size_ok = shot_bounce_time_ok && (size_ratio <= bullet_id_bounce_link_max_size_ratio_);
        if (!segment_size_ok && !shot_bounce_size_ok) continue;
        const bool segment_pool_gate_ok = segment_time_ok && segment_gap_ok && segment_size_ok;
        const bool shot_bounce_gate_ok = shot_bounce_time_ok && shot_bounce_gap_ok && shot_bounce_size_ok;

        double dir_norm = hypot2(seg.dir_x, seg.dir_y);
        double proj = 0.0, perp = dist;
        bool have_dir = dir_norm > 1e-6;
        double seg_dx = 0.0, seg_dy = 0.0;
        double forward_dir_cos = 1.0;
        if (have_dir) {
            seg_dx = seg.dir_x / dir_norm; seg_dy = seg.dir_y / dir_norm;
            proj = vx * seg_dx + vy * seg_dy;
            perp = hypot2(vx - proj * seg_dx, vy - proj * seg_dy);
            if (local_dir_ok) forward_dir_cos = local_dir_x * seg_dx + local_dir_y * seg_dy;
        }

        bool quality_ok = suffix_ok;
        std::string kind = "segment_pool";
        double score = -1e18;

        // P19: physical shot bounce-link.  Unlike P16/P17 forward/reverse-line
        // stitching, this branch intentionally accepts a large direction change
        // near the old segment endpoint.  It links the new segment to the same
        // physical bullet_id/shot, but hit_judge still decides HIT only from
        // body-mask contact evidence.
        if (shot_bounce_gate_ok && have_dir && local_dir_ok) {
            double dir_cos = local_dir_x * seg_dx + local_dir_y * seg_dy;
            bool turn_ok = (dir_cos <= bullet_id_bounce_link_max_dir_cos_);
            bool gap_ok = true;
            bool speed_ok = (local_speed >= bullet_id_bounce_link_min_speed_px_ms_) ||
                            (seg.speed_px_ms >= bullet_id_bounce_link_min_speed_px_ms_) ||
                            (implied_speed >= bullet_id_bounce_link_min_speed_px_ms_ * 0.60);
            bool disp_ok = (local_disp >= std::max(8.0, min_step_px * 2.0));
            bool size_ok = true;
            if (turn_ok && gap_ok && speed_ok && disp_ok && size_ok) {
                auto local_stats = slice_stats(local_pts);
                bool local_shape_ok = true;
                if (local_stats.has_value() && local_points >= 4) {
                    local_shape_ok = (local_stats->path_over_net <= 1.85 &&
                                      local_stats->sign_flip_ratio <= 0.45 &&
                                      local_stats->same_sign_ratio >= 0.55);
                }
                if (local_shape_ok) {
                    double angle_deg = std::acos(clampd(dir_cos, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
                    double bounce_score = 210.0
                        - dist * 1.15
                        - dt_ms * 0.18
                        + std::min(42.0, (angle_deg - bullet_id_bounce_link_min_turn_angle_deg_) * 0.55)
                        + std::min(28.0, std::max({local_speed, seg.speed_px_ms, implied_speed}) * 3.2)
                        - std::max(0.0, size_ratio - 1.0) * 6.0;
                    if (bounce_score > best_score) {
                        best_score = bounce_score;
                        best = &seg;
                        best_kind = "shot_bounce_link";
                    }
                }
            }
        }

        // From here down we are back in the legacy segment-pool stitch path.
        // P19 bounce-link has its own time/gap/size gate above, so do not let the
        // legacy P15/P16/P17 code run for candidates that only passed P19.
        if (!segment_pool_gate_ok) continue;

        if (sticky_handoff && have_dir && (!local_dir_ok || forward_dir_cos >= 0.74)) {
            double corridor = std::min(62.0, 24.0 + std::max(seg.speed_px_ms, local_speed) * dt_ms * 0.14);
            bool forward_ok = (proj >= -8.0 && perp <= corridor);
            bool speed_ok = (implied_speed >= std::max(0.38, segment_bounce_min_speed_px_ms_ * 0.80))
                         || (local_points >= 2 && local_disp >= 8.0 && local_speed >= 0.55);
            if (forward_ok && speed_ok) {
                quality_ok = true;
                kind = "raw_id_handoff_forward";
                score = 120.0 - perp * 1.4 - std::max(0.0, -proj) * 2.0 - 0.00010 * dt
                        - std::max(0.0, size_ratio - 1.0) * 8.0
                        + std::min(18.0, implied_speed * 2.2) + std::min(10.0, local_speed * 1.5);
            }
        }

        if (!quality_ok) continue;

        if (score < -1e17) {
            // 原 segment_pool：要求新 raw_id 自己已经形成弹道 suffix。
            double proj_ok = have_dir ? proj : dist;
            // P18: do not use bounce-style ID inheritance by default.  A new physical
            // shot may appear close to the previous one, but it should not inherit the
            // previous bullet_id unless it is a forward, same-direction continuation.
            if (have_dir && local_dir_ok && forward_dir_cos < 0.74) continue;
            if (have_dir && proj_ok < -8.0) continue;
            if (have_dir && perp > std::min(42.0, 22.0 + std::max(seg.speed_px_ms, local_speed) * dt_ms * 0.10)) continue;
            if (implied_speed < segment_bounce_min_speed_px_ms_) continue;
            score = -perp * 0.65 - std::max(0.0, -proj_ok) * 3.0 + (suffix_ok ? ok_stats->total_disp * 0.02 : local_disp * 0.01)
                    + seg.speed_px_ms - std::max(0.0, size_ratio - 1.0) * 2.0;
            kind = "segment_forward";
        }

        if (score > best_score) { best_score = score; best = &seg; best_kind = kind; }
    }

    if (!best) return std::nullopt;
    state.bullet_id = best->bullet_id;
    state.display_bullet_id = best->display_bullet_id >= 0 ? std::optional<int>(best->display_bullet_id) : std::optional<int>(best->bullet_id);
    state.birth_assign_kind = best_kind;
    // 直飞穿越黑色障碍物时仍是同一段弹道；反弹类才加 segment_index。
    if (best_kind == "segment_bounce" || best_kind == "shot_bounce_link") state.segment_index = best->segment_index + 1;
    else state.segment_index = best->segment_index;
    state.bullet_active = true;
    state.probation_passed = (best_kind == "raw_id_handoff_forward" || best_kind == "shot_bounce_link") ? false : true;
    if (best_kind == "shot_bounce_link") {
        double ldx = 0.0, ldy = 0.0, ldisp = 0.0;
        auto pts = state.points.to_vector();
        if (pts.size() >= 2) {
            ldx = pts.back().x - pts.front().x; ldy = pts.back().y - pts.front().y;
            ldisp = hypot2(ldx, ldy);
        }
        if (ldisp > 1e-6) state.segment_dir = std::make_pair(ldx / ldisp, ldy / ldisp);
        else state.segment_dir = std::make_pair(best->dir_x, best->dir_y);
        double oldn = hypot2(best->dir_x, best->dir_y);
        double newn = ldisp;
        double dir_cos = 1.0;
        if (oldn > 1e-6 && newn > 1e-6) dir_cos = (best->dir_x / oldn) * (ldx / newn) + (best->dir_y / oldn) * (ldy / newn);
        double angle_deg = std::acos(clampd(dir_cos, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
        double cur_x = state.last_cluster ? state.last_cluster->cx : (pts.empty() ? best->end_x : pts.back().x);
        double cur_y = state.last_cluster ? state.last_cluster->cy : (pts.empty() ? best->end_y : pts.back().y);
        state.shot_link_type = "bounce";
        state.shot_link_reason = "p19_cpp_shot_bounce_link";
        state.shot_link_parent_bullet_id = best->bullet_id;
        state.shot_link_parent_segment_index = best->segment_index;
        state.shot_link_parent_stable_track_id = best->stable_track_id;
        state.shot_link_parent_x = best->end_x;
        state.shot_link_parent_y = best->end_y;
        state.shot_link_gap_ms = (double)(ts - best->end_ts) / 1000.0;
        state.shot_link_gap_dist_px = hypot2(cur_x - best->end_x, cur_y - best->end_y);
        state.shot_link_angle_deg = angle_deg;
        state.shot_link_score = best_score;
        state.p12_owner_mode = "shot_bounce_link";
        state.p12_owner_bullet_id = best->bullet_id;
        state.p12_inherited_segment_index = best->segment_index;
        state.p12_owner_reason = "p19_cpp_shot_bounce_link";
    } else {
        state.segment_dir = std::make_pair(best->dir_x, best->dir_y);
        state.shot_link_type.clear();
        state.shot_link_reason.clear();
        state.shot_link_parent_bullet_id = -1;
        state.shot_link_parent_segment_index = -1;
        state.shot_link_parent_stable_track_id = -1;
        state.shot_link_parent_x = 0.0; state.shot_link_parent_y = 0.0;
        state.shot_link_gap_ms = 0.0; state.shot_link_gap_dist_px = 0.0;
        state.shot_link_angle_deg = 0.0; state.shot_link_score = 0.0;
    }
    state.segment_turn_warmup_until_ts = ts + segment_turn_warmup_us_;
    if (state.display_bullet_id.has_value()) displayed_bullet_ids_.insert(*state.display_bullet_id);
    return std::map<std::string,double>{{"bullet_id", (double)best->bullet_id}, {"display_bullet_id", (double)state.display_bullet_id.value_or(-1)}, {"segment_index", (double)state.segment_index}, {"source", best_kind == "shot_bounce_link" ? 4.0 : 1.0}, {"shot_link_angle_deg", state.shot_link_angle_deg}, {"shot_link_gap_ms", state.shot_link_gap_ms}, {"shot_link_gap_dist_px", state.shot_link_gap_dist_px}};
}

std::optional<std::map<std::string,double>> CppLineMotionFilter::try_confirmed_bullet_id_stitch(TrackState& state, int64_t ts) {
    if (!bullet_id_stitch_enabled) return std::nullopt;
    if (!state.last_cluster.has_value()) return std::nullopt;

    const Cluster& c = *state.last_cluster;
    const int self_tid = state.stable_track_id;

    auto local_pts = state.points.to_vector();
    int local_points = (int)local_pts.size();
    double local_dx = 0.0, local_dy = 0.0, local_dir_x = 0.0, local_dir_y = 0.0;
    bool local_dir_ok = false;
    double local_disp = 0.0;
    double local_speed = estimate_recent_speed_px_per_ms(state);
    if (local_points >= 2) {
        local_dx = local_pts.back().x - local_pts.front().x;
        local_dy = local_pts.back().y - local_pts.front().y;
        local_disp = hypot2(local_dx, local_dy);
        if (local_disp > 1e-6) {
            local_dir_x = local_dx / local_disp;
            local_dir_y = local_dy / local_disp;
            local_dir_ok = true;
        }
    }

    const double window_us = bullet_id_stitch_window_ms * 1000.0;
    const double active_window_us = bullet_id_stitch_active_max_dt_ms * 1000.0;

    double best_score = -1e18;
    int best_bullet = -1;
    int best_display = -1;
    int best_segment_index = 0;
    int best_from_track = -1;
    std::string best_source;

    auto consider = [&](int bullet_id, int display_id, int segment_index, int from_track_id,
                        int64_t anchor_ts, double ax, double ay, double dir_x, double dir_y,
                        double speed, double width, double height, const std::string& source,
                        bool allow_active_source) {
        if (bullet_id <= 0) return;
        if (anchor_ts < 0) return;
        int64_t dt = ts - anchor_ts;
        // Do not stitch two detections inside the exact same event slice; that is too
        // ambiguous and can merge simultaneous independent clusters.
        if (dt <= 0) return;
        double max_us = allow_active_source ? active_window_us : window_us;
        if ((double)dt > max_us) return;
        double dn = hypot2(dir_x, dir_y);
        if (dn <= 1e-6) return;
        dir_x /= dn; dir_y /= dn;

        double dt_ms = (double)dt / 1000.0;
        double pred_x = ax + dir_x * std::max(0.0, speed) * dt_ms * bullet_id_stitch_predict_gain;
        double pred_y = ay + dir_y * std::max(0.0, speed) * dt_ms * bullet_id_stitch_predict_gain;
        double vx = c.cx - ax;
        double vy = c.cy - ay;
        double proj = vx * dir_x + vy * dir_y;
        double perp = hypot2(vx - proj * dir_x, vy - proj * dir_y);
        double pred_dist = hypot2(c.cx - pred_x, c.cy - pred_y);
        double dist = hypot2(vx, vy);
        double implied_speed = dist / std::max(0.001, dt_ms);

        double sr = std::max({
            c.width / std::max(1.0, width), width / std::max(1.0, c.width),
            c.height / std::max(1.0, height), height / std::max(1.0, c.height)
        });
        if (sr > bullet_id_stitch_max_size_ratio) return;

        // The new segment must either already show motion in the same direction,
        // or be a very early 1-point segment lying tightly in the old corridor.
        double dir_cos = 1.0;
        if (local_dir_ok) {
            dir_cos = local_dir_x * dir_x + local_dir_y * dir_y;
            if (dir_cos < bullet_id_stitch_min_dir_cos) return;
        } else {
            if (perp > std::max(10.0, bullet_id_stitch_corridor_px * 0.55)) return;
        }

        double corridor = bullet_id_stitch_corridor_px + std::max(0.0, speed) * dt_ms * 0.10;
        corridor = std::min(std::max(corridor, bullet_id_stitch_corridor_px), bullet_id_stitch_corridor_px * 2.25);
        if (perp > corridor) return;

        // For true straight continuation, the new point normally appears ahead of the old
        // segment. Allow a small negative projection for model lag / bbox jitter only.
        const bool very_near_pred = pred_dist <= std::max(20.0, corridor * 0.90);
        const double min_forward = very_near_pred ? -18.0 : -8.0;
        if (proj < min_forward) return;

        // Reject slow human/edge drift. A very fresh 1-point segment is allowed only if
        // geometric prediction is tight; otherwise require local or implied speed.
        bool speed_ok = (local_speed >= bullet_id_stitch_min_speed_px_ms) ||
                        (implied_speed >= bullet_id_stitch_min_speed_px_ms) ||
                        (very_near_pred && dt_ms <= 28.0 && proj >= -4.0);
        if (!speed_ok) return;

        // If the local segment already has multiple points but almost no displacement,
        // it is more likely an obstacle residue than the flying bullet.
        if (local_points >= 3 && local_disp < std::max(5.0, min_step_px * 1.5)) return;

        double score = 240.0
            - perp * 3.2
            - pred_dist * 0.55
            - std::max(0.0, -proj) * 5.5
            - (sr - 1.0) * 8.0
            - (1.0 - dir_cos) * 35.0
            - dt_ms * 0.18
            + std::min(25.0, std::max(local_speed, implied_speed) * 3.5)
            + (source == "active_track" ? 14.0 : 0.0);

        if (score > best_score) {
            best_score = score;
            best_bullet = bullet_id;
            best_display = display_id > 0 ? display_id : bullet_id;
            best_segment_index = segment_index;
            best_from_track = from_track_id;
            best_source = source;
        }
    };


    // P17: reverse-line stitching.  The forward stitch above predicts from the old
    // bullet state to the new cluster.  If the old raw_id is sticky on a black
    // obstacle or the bullet is partially hidden, that forward prediction can be
    // unreliable.  Here the newly confirmed straight segment is used as the more
    // trustworthy observation: extend its motion line backwards and look for an old
    // physical bullet_id on that reverse line.
    auto consider_reverse = [&](int bullet_id, int display_id, int segment_index, int from_track_id,
                                int64_t anchor_ts, double ax, double ay, double dir_x, double dir_y,
                                double speed, double width, double height, const std::string& source,
                                bool allow_active_source) {
        if (!bullet_id_stitch_reverse_enabled) return;
        if (!local_dir_ok) return;
        if (local_points < 2) return;
        if (local_disp < bullet_id_stitch_reverse_min_local_disp_px) return;
        if (bullet_id <= 0 || anchor_ts < 0) return;
        int64_t dt = ts - anchor_ts;
        if (dt <= 0) return;
        double max_us = bullet_id_stitch_reverse_window_ms * 1000.0;
        if (allow_active_source) max_us = std::max(max_us, active_window_us);
        if ((double)dt > max_us) return;

        double old_dn = hypot2(dir_x, dir_y);
        if (old_dn <= 1e-6) return;
        dir_x /= old_dn; dir_y /= old_dn;

        double dir_cos = local_dir_x * dir_x + local_dir_y * dir_y;
        if (dir_cos < bullet_id_stitch_reverse_min_dir_cos) return;

        const Point5& start_pt = local_pts.front();
        double sx = start_pt.x;
        double sy = start_pt.y;
        double dt_ms = (double)dt / 1000.0;

        // Old anchor should lie behind the beginning/current point of the new segment
        // along the new segment direction.  A small negative allowance absorbs bbox /
        // model jitter, but a large negative value means the old bullet is actually
        // ahead of the new segment and must not be merged.
        double old_to_start_x = sx - ax;
        double old_to_start_y = sy - ay;
        double proj_to_start = old_to_start_x * local_dir_x + old_to_start_y * local_dir_y;
        double old_to_curr_x = c.cx - ax;
        double old_to_curr_y = c.cy - ay;
        double proj_to_curr = old_to_curr_x * local_dir_x + old_to_curr_y * local_dir_y;
        if (proj_to_start < -12.0 && proj_to_curr < -8.0) return;

        // Perpendicular distance from the old anchor to the reverse-extended new line.
        double avx = ax - sx;
        double avy = ay - sy;
        double aproj = avx * local_dir_x + avy * local_dir_y;
        double perp = hypot2(avx - aproj * local_dir_x, avy - aproj * local_dir_y);

        double corridor = bullet_id_stitch_reverse_corridor_px + std::max({0.0, local_speed, speed}) * dt_ms * 0.06;
        corridor = std::min(std::max(corridor, bullet_id_stitch_reverse_corridor_px), bullet_id_stitch_reverse_corridor_px * 2.4);
        if (perp > corridor) return;

        double sr = std::max({
            c.width / std::max(1.0, width), width / std::max(1.0, c.width),
            c.height / std::max(1.0, height), height / std::max(1.0, c.height)
        });
        if (sr > bullet_id_stitch_reverse_max_size_ratio) return;

        double dist = hypot2(old_to_curr_x, old_to_curr_y);
        double implied_speed = dist / std::max(0.001, dt_ms);
        bool speed_ok = (local_speed >= bullet_id_stitch_min_speed_px_ms) ||
                        (implied_speed >= bullet_id_stitch_min_speed_px_ms * 0.75) ||
                        (local_disp >= std::max(bullet_id_stitch_reverse_min_local_disp_px, min_total_displacement_px * 0.45));
        if (!speed_ok) return;

        // When the new segment already has enough points, it must itself remain nearly
        // straight.  This protects against merging human/obstacle edge motion that just
        // happens to be near the reverse line.
        if (local_points >= 4) {
            auto st_stats = slice_stats(local_pts);
            if (st_stats.has_value()) {
                if (st_stats->path_over_net > 1.65) return;
                if (st_stats->sign_flip_ratio > 0.35) return;
                if (st_stats->same_sign_ratio < 0.62) return;
            }
        }

        // Optional speed prediction penalty, deliberately soft because the visible gap
        // can include true occlusion and sticky raw_id lag.
        double vref = std::max({local_speed, speed, bullet_id_stitch_min_speed_px_ms});
        double expected_gap = vref * dt_ms * 0.70;
        double pred_gap_err = std::abs(std::max(0.0, proj_to_curr) - expected_gap);

        double score = 270.0
            - perp * 3.8
            - std::max(0.0, -proj_to_start) * 7.0
            - (1.0 - dir_cos) * 46.0
            - (sr - 1.0) * 7.0
            - dt_ms * 0.11
            - std::min(80.0, pred_gap_err * 0.18)
            + std::min(30.0, std::max(local_speed, implied_speed) * 4.0)
            + (source == "reverse_active_track" ? 18.0 : 0.0);

        if (score > best_score) {
            best_score = score;
            best_bullet = bullet_id;
            best_display = display_id > 0 ? display_id : bullet_id;
            best_segment_index = segment_index;
            best_from_track = from_track_id;
            best_source = source;
        }
    };

    // 1) Active/confirmed tracks. This is the key case for obstacle boundary: the old
    // raw_id may still be alive/sticky, while the true bullet has already become a new raw_id.
    for (const auto& kv : tracks_) {
        const TrackState& src = kv.second;
        if (src.stable_track_id == self_tid) continue;
        if (!src.bullet_id.has_value()) continue;
        if (!(src.bullet_active || src.confirmed_once || src.display_bullet_id.has_value())) continue;
        // Only inherit from a physical bullet that has already been displayed/confirmed.
        // This prevents two newly-born noise tracks from immediately sharing an ID.
        if (!(src.display_bullet_id.has_value() || displayed_bullet_ids_.count(*src.bullet_id) > 0 || src.probation_passed)) continue;
        int display_id = src.display_bullet_id.value_or(*src.bullet_id);
        if (display_id <= 0) continue;

        double ax = 0.0, ay = 0.0;
        int64_t anchor_ts = -1;
        if (src.model_ready && src.model_last_ts >= 0) {
            ax = src.model_x; ay = src.model_y; anchor_ts = src.model_last_ts;
        } else if (src.last_cluster.has_value()) {
            ax = src.last_cluster->cx; ay = src.last_cluster->cy; anchor_ts = ts - (int64_t)std::max(0, src.miss_count) * (int64_t)4000;
        } else {
            continue;
        }
        double dx = 0.0, dy = 0.0;
        // P18: for ID stitching, use the locked/segment direction first.
        // The model velocity can be polluted when Spatter raw_id sticks on a black obstacle,
        // and that polluted direction was the main reason a later opposite-direction bullet
        // could inherit the previous bullet_id.
        if (src.segment_dir.has_value()) {
            dx = src.segment_dir->first; dy = src.segment_dir->second;
        } else {
            auto td = track_direction(src);
            if (td.has_value()) {
                dx = td->first; dy = td->second;
            } else if (src.model_ready && hypot2(src.model_vx, src.model_vy) > 1e-6) {
                dx = src.model_vx; dy = src.model_vy;
            } else {
                continue;
            }
        }
        double speed = src.model_ready ? src.model_speed_px_ms : estimate_recent_speed_px_per_ms(src);
        double w = src.last_cluster ? std::max(1.0, src.last_cluster->width) : std::max(1.0, c.width);
        double h = src.last_cluster ? std::max(1.0, src.last_cluster->height) : std::max(1.0, c.height);
        consider(*src.bullet_id, display_id, src.segment_index, src.stable_track_id, anchor_ts, ax, ay, dx, dy, speed, w, h, "active_track", true);
        consider_reverse(*src.bullet_id, display_id, src.segment_index, src.stable_track_id, anchor_ts, ax, ay, dx, dy, speed, w, h, "reverse_active_track", true);
    }

    // 2) Recently terminated/pooled segments are still useful when the old track was already killed.
    for (const auto& seg : recent_segments_) {
        consider(seg.bullet_id, seg.display_bullet_id, seg.segment_index, seg.stable_track_id,
                 seg.end_ts, seg.end_x, seg.end_y, seg.dir_x, seg.dir_y, seg.speed_px_ms,
                 seg.width, seg.height, "recent_segment", false);
        consider_reverse(seg.bullet_id, seg.display_bullet_id, seg.segment_index, seg.stable_track_id,
                 seg.end_ts, seg.end_x, seg.end_y, seg.dir_x, seg.dir_y, seg.speed_px_ms,
                 seg.width, seg.height, "reverse_segment", false);
    }
    for (const auto& term : recent_terminated_) {
        consider(term.bullet_id, term.display_bullet_id, term.segment_index, term.stable_track_id,
                 term.ts, term.x, term.y, term.dir_x, term.dir_y, term.speed_px_ms,
                 term.width, term.height, "recent_terminated", false);
        consider_reverse(term.bullet_id, term.display_bullet_id, term.segment_index, term.stable_track_id,
                 term.ts, term.x, term.y, term.dir_x, term.dir_y, term.speed_px_ms,
                 term.width, term.height, "reverse_terminated", false);
    }

    if (best_bullet <= 0) return std::nullopt;
    state.bullet_id = best_bullet;
    state.display_bullet_id = best_display > 0 ? std::optional<int>(best_display) : std::optional<int>(best_bullet);
    if (best_source == "active_track") state.birth_assign_kind = "bullet_id_stitch_active";
    else if (best_source == "recent_segment") state.birth_assign_kind = "bullet_id_stitch_segment";
    else if (best_source == "recent_terminated") state.birth_assign_kind = "bullet_id_stitch_terminated";
    else if (best_source == "reverse_active_track") state.birth_assign_kind = "bullet_id_stitch_reverse_active";
    else if (best_source == "reverse_segment") state.birth_assign_kind = "bullet_id_stitch_reverse_segment";
    else if (best_source == "reverse_terminated") state.birth_assign_kind = "bullet_id_stitch_reverse_terminated";
    else state.birth_assign_kind = "bullet_id_stitch";
    state.segment_index = best_segment_index;
    state.bullet_active = true;
    // Keep a short probation after stitching so a wrong merge can still be terminated by
    // the existing maintain/probation gates. Do not force probation_passed=true here.
    state.probation_passed = false;
    state.p12_owner_mode = state.birth_assign_kind;
    state.p12_owner_bullet_id = best_bullet;
    state.p12_inherited_segment_index = best_segment_index;
    state.p12_owner_reason = (best_source.rfind("reverse_", 0) == 0) ? "p17_reverse_line_stitch" : "p16_display_id_stitch";
    if (state.display_bullet_id.has_value()) displayed_bullet_ids_.insert(*state.display_bullet_id);
    double source_code = (best_source.rfind("reverse_", 0) == 0) ? 3.0 : 2.0;
    return std::map<std::string,double>{{"bullet_id", (double)best_bullet}, {"display_bullet_id", (double)state.display_bullet_id.value_or(-1)}, {"segment_index", (double)state.segment_index}, {"source", source_code}, {"from_track", (double)best_from_track}, {"score", best_score}};
}

std::optional<std::map<std::string,double>> CppLineMotionFilter::try_ghost_capture_new_track(TrackState& state, int64_t ts) {
    if (!ghost_capture_enabled) return std::nullopt;
    auto a = try_segment_pool_capture_new_track(state, ts);
    if (a.has_value()) {
        state.birth_assign_kind = "ghost_capture_segment";
        return a;
    }
    auto b = try_reuse_recent_terminated_bullet(state, ts);
    if (b.has_value()) {
        state.birth_assign_kind = "ghost_capture";
        return b;
    }
    return std::nullopt;
}
bool CppLineMotionFilter::is_suspected_continuation(const TrackState& state, int64_t ts) const {
    if (!state.last_cluster.has_value()) return false;
    const Cluster& c = *state.last_cluster;
    for (const auto& seg : recent_segments_) {
        int64_t dt = ts - seg.end_ts;
        if (dt < 0 || dt > recent_segment_pool_window_us_) continue;
        double dist = hypot2(c.cx - seg.end_x, c.cy - seg.end_y);
        if (dist <= segment_bounce_adaptive_max_px_) return true;
    }
    for (const auto& item : recent_terminated_) {
        int64_t dt = ts - item.ts;
        if (dt < 0 || dt > ghost_reactivate_window_us_) continue;
        if (hypot2(c.cx - item.x, c.cy - item.y) <= ghost_base_distance_px_) return true;
    }
    return false;
}
std::vector<Point5> CppLineMotionFilter::select_confirmed_backfill_points(TrackState& state, int64_t current_ts) {
    std::vector<Point5> out;
    if (!confirmed_backfill_enabled_) return out;
    std::vector<Point3> hist;
    std::set<int64_t> seen;
    for (const auto& p : state.pre_confirm_points.raw()) {
        if (p.ts > current_ts) continue;
        if (seen.insert(p.ts).second) hist.push_back(p);
    }
    std::sort(hist.begin(), hist.end(), [](const Point3& a, const Point3& b){ return a.ts < b.ts; });
    hist.erase(std::remove_if(hist.begin(), hist.end(), [&](const Point3& p){ return current_ts - p.ts < 0 || current_ts - p.ts > confirmed_backfill_max_age_us_; }), hist.end());
    if ((int)hist.size() > confirmed_backfill_max_points_) hist.erase(hist.begin(), hist.end() - confirmed_backfill_max_points_);
    if ((int)hist.size() < confirmed_backfill_min_points_) return out;
    std::map<int64_t, std::pair<double,double>> wh;
    for (const auto& p : state.points.raw()) wh[p.ts] = {std::max(1.0,p.w), std::max(1.0,p.h)};
    double cur_w = state.last_cluster ? std::max(1.0, state.last_cluster->width) : 1.0;
    double cur_h = state.last_cluster ? std::max(1.0, state.last_cluster->height) : 1.0;
    int end_idx = (int)hist.size() - 1;
    int min_start = std::max(0, end_idx - confirmed_backfill_max_points_ + 1);
    double best_score = -1e18;
    std::vector<Point5> best;
    for (int start = min_start; start <= end_idx - confirmed_backfill_min_points_ + 1; ++start) {
        std::vector<Point5> pseudo;
        for (int i = start; i <= end_idx; ++i) {
            auto it = wh.find(hist[i].ts);
            double w = it == wh.end() ? cur_w : it->second.first;
            double h = it == wh.end() ? cur_h : it->second.second;
            pseudo.push_back(Point5{hist[i].x, hist[i].y, w, h, hist[i].ts});
        }
        auto stats = slice_stats(pseudo);
        if (!stats.has_value()) continue;
        double speed = estimate_speed_from_points(pseudo);
        int64_t duration = pseudo.back().ts - pseudo.front().ts;
        if (duration < 3000) continue;
        bool fail = false;
        if (speed < confirmed_backfill_min_speed_px_ms_) fail = true;
        if (stats->valid_step_count < std::max(2, confirmed_backfill_min_points_ - 1)) fail = true;
        if (stats->path_over_net > confirmed_backfill_max_path_over_net_) fail = true;
        if (stats->max_offset > max_line_offset_px * confirmed_backfill_offset_ratio_) fail = true;
        if (stats->direction_ok_ratio < confirmed_backfill_min_direction_ok_) fail = true;
        if (stats->same_sign_ratio < confirmed_backfill_min_same_sign_) fail = true;
        if (stats->sign_flip_ratio > confirmed_backfill_max_sign_flip_) fail = true;
        if (fail) continue;
        double score = pseudo.size()*0.20 + std::min(3.0, speed) + std::min(2.0, stats->total_disp/std::max(1.0,min_total_displacement_px)) + std::max(0.0, 2.0 - stats->path_over_net) + std::max(0.0, 2.0 - stats->max_offset/std::max(1.0,max_line_offset_px));
        if (score > best_score) { best_score = score; best = pseudo; }
    }
    return best;
}
void CppLineMotionFilter::apply_confirmed_backfill_points(TrackState& state, const std::vector<Point5>& pts) {
    if (pts.empty()) return;
    std::vector<Point3> merged;
    std::set<int64_t> seen;
    for (const auto& p : pts) if (seen.insert(p.ts).second) merged.push_back(Point3{p.x,p.y,p.ts});
    for (const auto& p : state.model_points.raw()) if (seen.insert(p.ts).second) merged.push_back(p);
    std::sort(merged.begin(), merged.end(), [](const Point3& a, const Point3& b){ return a.ts < b.ts; });
    if (merged.empty()) return;
    std::vector<Point3> model = merged;
    if ((int)model.size() > std::max(80, draw_history_len)) model.erase(model.begin(), model.end() - std::max(80, draw_history_len));
    state.model_points.set_maxlen(80); state.model_points.assign(model);
    std::vector<Point3> disp = merged;
    if ((int)disp.size() > draw_history_len) disp.erase(disp.begin(), disp.end() - draw_history_len);
    state.display_points.set_maxlen(draw_history_len); state.display_points.assign(disp);
    std::vector<Point3> cb = merged;
    if ((int)cb.size() > 32) cb.erase(cb.begin(), cb.end() - 32);
    state.confirmed_backfill_points.assign(cb);
}
std::vector<DebugRow> CppLineMotionFilter::build_confirmed_backfill_debug_rows(TrackState& state, int64_t raw_id, int stable_track_id, const Cluster& current_cluster, const std::vector<Point5>& pts, int64_t current_ts) {
    std::vector<DebugRow> rows;
    if (pts.empty() || !state.bullet_id.has_value()) return rows;
    int bullet_id = *state.bullet_id;
    int disp_id = state.display_bullet_id.value_or(-1);
    double base_w = std::max(1.0, current_cluster.width), base_h = std::max(1.0, current_cluster.height);
    for (const auto& p : pts) {
        if (p.ts >= current_ts) continue;
        if (p.ts <= state.confirmed_backfill_last_emit_ts) continue;
        double ww = std::max(1.0, p.w > 0 ? p.w : base_w);
        double hh = std::max(1.0, p.h > 0 ? p.h : base_h);
        DebugRow r;
        r.integer["timestamp"] = p.ts; r.integer["raw_id"] = raw_id; r.integer["stable_track_id"] = stable_track_id;
        r.text["assign_kind"] = "confirmed_backfill"; r.text["phase"] = "confirmed_backfill"; r.text["reject_reason"] = "confirmed_backfill"; r.text["reuse_source"] = "confirmed_backfill";
        r.numeric["x"] = p.x - 0.5 * ww; r.numeric["y"] = p.y - 0.5 * hh; r.numeric["width"] = ww; r.numeric["height"] = hh;
        r.numeric["cx"] = p.x; r.numeric["cy"] = p.y; r.numeric["obs_x"] = p.x; r.numeric["obs_y"] = p.y; r.numeric["model_x"] = p.x; r.numeric["model_y"] = p.y; r.numeric["pred_x"] = p.x; r.numeric["pred_y"] = p.y;
        r.numeric["model_residual_px"] = 0.0; r.integer["model_obs_used"] = 1; r.integer["model_outlier"] = 0; r.text["model_update_mode"] = "confirmed_backfill";
        r.integer["keep_now"] = 1; r.integer["confirmed_now"] = 1; r.integer["accepted_now"] = 1; r.integer["bullet_id"] = bullet_id; r.integer["display_bullet_id"] = disp_id;
        r.integer["just_assigned_bullet"] = 0; r.text["birth_assign_kind"] = state.birth_assign_kind; r.integer["bullet_active"] = 1; r.integer["probation_passed"] = state.probation_passed ? 1 : 0;
        r.integer["segment_index"] = state.segment_index; r.integer["confirmed_backfill"] = 1; r.integer["confirmed_backfill_n"] = (int64_t)pts.size(); r.integer["recent_segment_pool_size"] = (int64_t)recent_segments_.size();
        rows.push_back(std::move(r));
        state.confirmed_backfill_last_emit_ts = std::max<int64_t>(state.confirmed_backfill_last_emit_ts, p.ts);
    }
    return rows;
}
double CppLineMotionFilter::association_cost(const TrackState& state, const Cluster& cluster, int64_t ts) const {
    (void)ts;
    if (state.points.empty()) return std::numeric_limits<double>::quiet_NaN();
    auto pred = predict_center(state, cluster.raw_id >= 0 ? state.points.back().ts : ts);
    double dist = hypot2(cluster.cx - pred.first, cluster.cy - pred.second);
    double allow_dist = ac_.max_distance_px * (1.0 + 0.35 * std::max(0, state.miss_count));
    if (state.bullet_active) allow_dist *= 1.40;
    if (dist > allow_dist) return std::numeric_limits<double>::quiet_NaN();
    double size_ratio = 1.0;
    if (state.last_cluster.has_value()) {
        const auto& lc = *state.last_cluster;
        double last_w = std::max(1.0, lc.width);
        double last_h = std::max(1.0, lc.height);
        double cur_w = std::max(1.0, cluster.width);
        double cur_h = std::max(1.0, cluster.height);
        size_ratio = std::max({cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h});
        double allowed_sr = ac_.max_size_ratio * (state.bullet_active ? 1.45 : 1.0);
        if (size_ratio > allowed_sr) return std::numeric_limits<double>::quiet_NaN();
    }
    double cost = dist + (size_ratio - 1.0) * 4.0;
    auto direction = track_direction(state);
    if (direction.has_value() && state.points.size() > 0) {
        const auto& last_pt = state.points.back();
        double ox = cluster.cx - last_pt.x;
        double oy = cluster.cy - last_pt.y;
        double on = hypot2(ox, oy);
        if (on >= std::max(1.0, min_step_px * 0.4)) {
            double cos_val = dot2(ox / std::max(on, 1e-6), oy / std::max(on, 1e-6), direction->first, direction->second);
            cos_val = std::max(-1.0, std::min(1.0, cos_val));
            double penalty_scale = state.bullet_active ? 0.30 : 1.0;
            if (!state.bullet_active && cos_val < -0.15) {
                cost += (1.0 - std::abs(cos_val)) * (ac_.direction_penalty_px * 3.0);
            } else {
                cost += (1.0 - std::abs(cos_val)) * (ac_.direction_penalty_px * penalty_scale);
            }
        }
    }
    return cost;
}
bool CppLineMotionFilter::direct_raw_id_association_ok(const TrackState& state, const Cluster& c, int64_t ts) const {
    // P13: Spatter raw_id 粘背景保护。只拦截已确认/曾确认子弹的 raw_id 直连，
    // 被拦截的 cluster 仍可进入空间 assoc 或新 track。
    if (!raw_id_sticky_guard_enabled) return true;
    if (!(state.bullet_active || state.confirmed_once || state.bullet_id.has_value() || state.display_bullet_id.has_value())) return true;
    if (state.points.size() < 2) return true;

    double pred_x = 0.0, pred_y = 0.0;
    double dir_x = 0.0, dir_y = 0.0;
    bool have_dir = false;
    double speed = 0.0;
    if (state.model_ready) {
        auto pred = predict_model_xy(state, ts);
        pred_x = pred.first; pred_y = pred.second;
        speed = hypot2(state.model_vx, state.model_vy);
        if (speed > 1e-6) { dir_x = state.model_vx / speed; dir_y = state.model_vy / speed; have_dir = true; }
    } else {
        auto pred = predict_center(state, ts);
        pred_x = pred.first; pred_y = pred.second;
        auto dir = track_direction(state);
        if (dir.has_value()) { dir_x = dir->first; dir_y = dir->second; have_dir = true; }
        speed = estimate_recent_speed_px_per_ms(state);
    }

    double residual = hypot2(c.cx - pred_x, c.cy - pred_y);
    const auto& last_pt = state.points.back();
    double sx = c.cx - last_pt.x;
    double sy = c.cy - last_pt.y;
    double step_len = hypot2(sx, sy);

    if (speed >= raw_id_sticky_guard_min_speed_px_ms && step_len <= raw_id_sticky_guard_static_step_px) return false;
    double residual_limit = std::max(8.0, raw_id_sticky_guard_residual_px);
    if (residual > residual_limit) return false;

    if (have_dir && speed >= raw_id_sticky_guard_min_speed_px_ms) {
        double fwd = sx * dir_x + sy * dir_y;
        if (fwd < -std::abs(raw_id_sticky_guard_backward_px)) return false;
        if (step_len <= std::max(raw_id_sticky_guard_static_step_px, min_step_px * 0.75) && fwd < min_step_px * 0.20) return false;
    }

    if (state.last_cluster.has_value()) {
        const auto& lc = *state.last_cluster;
        double last_w = std::max(1.0, lc.width);
        double last_h = std::max(1.0, lc.height);
        double cur_w = std::max(1.0, c.width);
        double cur_h = std::max(1.0, c.height);
        double sr = std::max({cur_w / last_w, last_w / cur_w, cur_h / last_h, last_h / cur_h});
        if (sr > ac_.max_size_ratio * 1.45) return false;
    }

    return true;
}

std::map<int,int> CppLineMotionFilter::assign_clusters(const std::vector<Cluster>& clusters, int64_t ts) {
    std::map<int,int> assignments;
    std::set<int> matched_track_ids;
    std::vector<Cluster> pending_clusters;
    for (const auto& c : clusters) {
        int64_t raw_id = c.raw_id;
        auto it = raw_to_track_.find(raw_id);
        if (it != raw_to_track_.end() && tracks_.find(it->second) != tracks_.end() && !matched_track_ids.count(it->second)
            && direct_raw_id_association_ok(tracks_.at(it->second), c, ts)) {
            assignments[(int)raw_id] = it->second;
            matched_track_ids.insert(it->second);
        } else {
            // P14: raw_id 粘背景导致同一颗子弹换 raw_id 时，把旧 bullet 的最后可信模型段
            // 放入 recent_segment/owner 池，后续新 raw_id 可以继承同一个 bullet_id。
            if (it != raw_to_track_.end() && tracks_.find(it->second) != tracks_.end() && !matched_track_ids.count(it->second)) {
                TrackState& st = tracks_.at(it->second);
                if (st.bullet_active || st.confirmed_once || st.bullet_id.has_value() || st.display_bullet_id.has_value()) {
                    push_recent_segment(st, ts, "raw_id_sticky_handoff");
                    push_occlusion_owner(st, ts, "raw_id_sticky_handoff");
                }
            }
            pending_clusters.push_back(c);
        }
    }
    struct Candidate { double cost; int64_t raw_id; int stable_track_id; };
    std::vector<Candidate> candidates;
    for (const auto& c : pending_clusters) {
        for (const auto& kv : tracks_) {
            int tid = kv.first;
            if (matched_track_ids.count(tid)) continue;
            double cost = association_cost(kv.second, c, ts);
            if (!std::isnan(cost)) candidates.push_back({cost, c.raw_id, tid});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b){ return a.cost < b.cost; });
    std::set<int64_t> matched_raw_ids;
    for (const auto& cand : candidates) {
        if (matched_raw_ids.count(cand.raw_id) || matched_track_ids.count(cand.stable_track_id)) continue;
        assignments[(int)cand.raw_id] = cand.stable_track_id;
        matched_raw_ids.insert(cand.raw_id);
        matched_track_ids.insert(cand.stable_track_id);
    }
    for (const auto& c : clusters) {
        if (assignments.count((int)c.raw_id)) continue;
        TrackState& st = create_state();
        assignments[(int)c.raw_id] = st.stable_track_id;
        matched_track_ids.insert(st.stable_track_id);
    }
    return assignments;
}
bool CppLineMotionFilter::try_promote_display_early(TrackState& state, const SliceStats* full_stats, const SliceStats* recent_stats, int64_t ts) {
    (void)full_stats; (void)ts;
    if (state.display_bullet_id.has_value() || !state.bullet_active || state.probation_passed) return false;
    const SliceStats* st = recent_stats;
    if (!st) {
        auto local = compute_stats(state.points);
        if (!local.has_value()) return false;
        double speed = estimate_recent_speed_px_per_ms(state);
        bool ok = local->total_disp >= std::max(10.0, min_total_displacement_px * 0.50)
            && local->max_offset <= max_line_offset_px * 1.50
            && local->direction_ok_ratio >= 0.68
            && local->same_sign_ratio >= 0.68
            && local->sign_flip_ratio <= 0.38
            && local->path_over_net <= 2.35
            && local->valid_step_count >= 1
            && speed >= std::max(0.35, trigger_min_avg_speed_px_per_ms * 0.55);
        if (!ok) return false;
    } else {
        double speed = estimate_recent_speed_px_per_ms(state);
        bool ok = st->total_disp >= std::max(10.0, min_total_displacement_px * 0.50)
            && st->max_offset <= max_line_offset_px * 1.50
            && st->direction_ok_ratio >= 0.68
            && st->same_sign_ratio >= 0.68
            && st->sign_flip_ratio <= 0.38
            && st->path_over_net <= 2.35
            && st->valid_step_count >= 1
            && speed >= std::max(0.35, trigger_min_avg_speed_px_per_ms * 0.55);
        if (!ok) return false;
    }
    promote_display_id(state);
    return true;
}
std::map<std::string,double> CppLineMotionFilter::post_assign_stats(TrackState& state, int64_t ts) {
    (void)ts;
    if (state.bullet_assign_ts < 0) return {};
    if (state.model_ready && state.model_points.size() >= 2) {
        std::vector<Point5> pseudo;
        for (const auto& p : state.model_points.raw()) if (p.ts >= state.bullet_assign_ts) pseudo.push_back(Point5{p.x,p.y,1.0,1.0,p.ts});
        if (pseudo.size() >= 2) { auto s = slice_stats(pseudo); if (s) return stats_to_map(*s); }
    }
    std::vector<Point5> pts;
    for (const auto& p : state.points.raw()) if (p.ts >= state.bullet_assign_ts) pts.push_back(p);
    if (pts.size() < 2) return {};
    auto st = slice_stats(pts);
    return st ? stats_to_map(*st) : std::map<std::string,double>{};
}
int CppLineMotionFilter::required_trigger_streak(const TrackState& state, bool ballistic_rearm) const {
    if (ballistic_rearm) return std::min(bullet_min_output_streak, ballistic_rearm_required_streak_);
    int required = bullet_min_output_streak;
    auto pts = last_n_point5(state.points, recent_window_points_);
    auto st = slice_stats(pts);
    if (st.has_value()) {
        bool very_clean = st->direction_ok_ratio >= 0.95 && st->same_sign_ratio >= 0.95 && st->max_offset <= std::min(1.5, max_line_offset_px * 0.45) && st->total_disp >= std::max(35.0, min_total_displacement_px * 1.2) && st->path_over_net <= 1.15 && st->sign_flip_ratio <= 0.05;
        if (very_clean) required = std::min(required, 3);
    }
    return required;
}
std::optional<SliceStats> CppLineMotionFilter::select_post_outlier_rearm_suffix_stats(const TrackState& state, int64_t ts) const {
    if (!post_outlier_rearm_enabled_) return std::nullopt;
    if (!state.last_bullet_id.has_value() || state.last_terminated_ts < 0) return std::nullopt;
    if (post_outlier_rearm_reasons_.count(state.last_terminated_reason) == 0) return std::nullopt;
    int64_t dt = ts - state.last_terminated_ts;
    if (dt < post_outlier_rearm_min_dt_us_ || dt > post_outlier_rearm_window_us_) return std::nullopt;
    return select_ballistic_rearm_suffix_stats(state, ts);
}
void CppLineMotionFilter::reset_points_to_recent_suffix_for_rearm(TrackState& state, int64_t ts) {
    (void)ts;
    if (!post_outlier_rearm_reset_points_) return;
    auto pts = state.points.to_vector();
    if (pts.empty()) return;
    int k = std::max(ballistic_rearm_window_min_, std::min(state.post_outlier_rearm_last_window > 0 ? state.post_outlier_rearm_last_window : ballistic_rearm_window_max_, ballistic_rearm_window_max_));
    if ((int)pts.size() < k) k = (int)pts.size();
    if (k < 2) return;
    std::vector<Point5> suffix(pts.end() - k, pts.end());
    state.points.set_maxlen(history_len); state.points.assign(suffix);
    std::vector<Point3> p3;
    std::vector<double> widths, heights;
    for (const auto& p : suffix) { p3.push_back(Point3{p.x,p.y,p.ts}); widths.push_back(p.w); heights.push_back(p.h); }
    state.pre_confirm_points.set_maxlen(pre_confirm_history_len); state.pre_confirm_points.assign(p3);
    state.obs_points.set_maxlen(48); state.obs_points.assign(p3);
    state.display_points.clear(); state.model_points.clear();
    state.model_ready = false; state.model_last_ts = -1; state.model_update_mode = "post_outlier_rearm_reset";
    state.recent_widths.assign(widths); state.recent_heights.assign(heights);
}
std::optional<SliceStats> CppLineMotionFilter::select_ballistic_rearm_suffix_stats(const TrackState& state, int64_t ts) const {
    (void)ts;
    if (!ballistic_rearm_enabled_) return std::nullopt;
    auto pts_all = state.points.to_vector();
    if ((int)pts_all.size() < ballistic_rearm_window_min_) return std::nullopt;
    std::optional<SliceStats> best;
    double best_score = -1e18;
    for (int k = ballistic_rearm_window_min_; k <= ballistic_rearm_window_max_; ++k) {
        if ((int)pts_all.size() < k) continue;
        std::vector<Point5> pts(pts_all.end()-k, pts_all.end());
        auto stats = slice_stats(pts);
        if (!stats.has_value()) continue;
        double speed = estimate_speed_from_points(pts);
        std::vector<double> areas, longs;
        for (const auto& p : pts) { double w=std::max(1.0,p.w), h=std::max(1.0,p.h); areas.push_back(w*h); longs.push_back(std::max(w,h)); }
        double area_med = median_value(areas, 0.0), long_med = median_value(longs, 0.0), area_last=areas.back(), long_last=longs.back();
        double disp_th = std::max(8.0, min_total_displacement_px * ballistic_rearm_disp_ratio_);
        double offset_th = max_line_offset_px * ballistic_rearm_offset_ratio_;
        bool fail = stats->total_disp < disp_th || speed < trigger_min_avg_speed_px_per_ms || stats->valid_step_count < ballistic_rearm_min_valid_steps_ || stats->path_over_net > ballistic_rearm_max_path_over_net_ || stats->max_offset > offset_th || stats->direction_ok_ratio < ballistic_rearm_min_direction_ok_ratio_ || stats->same_sign_ratio < ballistic_rearm_min_same_sign_ratio_ || stats->sign_flip_ratio > ballistic_rearm_max_sign_flip_ratio_ || area_med > ballistic_rearm_max_bbox_area_ || long_med > ballistic_rearm_max_bbox_long_side_ || area_last > ballistic_rearm_max_bbox_area_ || long_last > ballistic_rearm_max_bbox_long_side_;
        if (fail) continue;
        double score = std::min(2.0, stats->total_disp/std::max(1e-6,disp_th)) + std::min(2.0, speed/std::max(1e-6,trigger_min_avg_speed_px_per_ms)) + std::min(1.5, (double)stats->valid_step_count/std::max(1,ballistic_rearm_min_valid_steps_)) + std::min(1.5, ballistic_rearm_max_path_over_net_/std::max(1e-6,stats->path_over_net)) + std::min(1.5, offset_th/std::max(1e-6,stats->max_offset));
        if (score > best_score) { best_score=score; best=stats; }
    }
    return best;
}
std::map<std::string,double> CppLineMotionFilter::evaluate_trigger(TrackState& state, const Cluster& cluster, int64_t ts) {
    (void)cluster;
    std::map<std::string,double> out;
    out["keep_now"] = 0; out["trigger_ok"] = 0; out["maintain_ok"] = 0; out["confirmed_now"] = 0;
    out["line_ok"] = 0; out["direction_ok"] = 0; out["motion_ok"] = 0; out["recent_geom_ok"] = 0; out["recent_motion_ok"] = 0;
    out["trigger_disp_thresh"] = min_total_displacement_px; out["compact_ok"] = 0;
    out["rearm_mode"] = 0; out["ballistic_rearm"] = 0; out["ballistic_rearm_window"] = 0; out["ballistic_rearm_speed_px_ms"] = 0;
    out["post_outlier_rearm"] = 0; out["post_outlier_rearm_window"] = 0; out["post_outlier_rearm_dt_us"] = 0; out["required_trigger_streak_override"] = 0;
    auto full_opt = compute_stats(state.points);
    auto recent_opt = slice_stats(last_n_point5(state.points, recent_window_points_));
    if (!full_opt || !recent_opt) { out["phase_code"] = 0; return out; }
    const auto& full = *full_opt; auto recent = *recent_opt;
    if ((int)state.points.size() <= bootstrap_frames) { out["phase_code"] = 1; return out; }

    int64_t track_age_us = ts - state.first_seen_ts;
    bool suspected_continuation = is_suspected_continuation(state, ts);
    bool late_trigger_veto = (!state.confirmed_once && !state.last_bullet_id.has_value() && track_age_us > late_trigger_max_age_us_ && !suspected_continuation);
    bool rearm_mode = false, ballistic_rearm = false, post_outlier_rearm = false;
    int rearm_window = 0; double rearm_speed = 0.0; int64_t post_dt_us = 0;

    auto por = select_post_outlier_rearm_suffix_stats(state, ts);
    if (por) {
        post_outlier_rearm = true; ballistic_rearm = true; rearm_mode = true; recent = *por; late_trigger_veto = false;
        rearm_window = (int)recent.n_points; rearm_speed = estimate_speed_from_points(last_n_point5(state.points, recent.n_points));
        post_dt_us = state.post_outlier_rearm_last_ts >= 0 ? ts - state.post_outlier_rearm_last_ts : 0;
    }
    if (!post_outlier_rearm && late_trigger_veto) {
        rearm_mode = recent.direction_ok_ratio >= 0.90 && recent.same_sign_ratio >= 0.90 && recent.path_over_net <= 1.30 &&
                     recent.max_offset <= max_line_offset_px && recent.total_disp >= std::max(12.0, min_total_displacement_px * 0.60) && recent.sign_flip_ratio <= 0.10;
        if (rearm_mode) late_trigger_veto = false;
        else {
            auto br = select_ballistic_rearm_suffix_stats(state, ts);
            if (br) { ballistic_rearm = true; rearm_mode = true; recent = *br; late_trigger_veto = false; rearm_window = recent.n_points; rearm_speed = estimate_speed_from_points(last_n_point5(state.points, recent.n_points)); }
        }
    }

    bool line_ok, direction_ok;
    if (ballistic_rearm) {
        line_ok = recent.max_offset <= max_line_offset_px * ballistic_rearm_offset_ratio_;
        direction_ok = recent.direction_ok_ratio >= ballistic_rearm_min_direction_ok_ratio_ && recent.same_sign_ratio >= ballistic_rearm_min_same_sign_ratio_;
    } else {
        line_ok = recent.max_offset <= trigger_max_recent_offset_px_;
        direction_ok = recent.direction_ok_ratio >= trigger_min_dir_ratio_ && recent.same_sign_ratio >= trigger_min_same_sign_;
    }
    bool continuation_geom_ok = full.max_offset <= max_line_offset_px * 1.2;
    bool effective_suspected = suspected_continuation && continuation_geom_ok;
    const SliceStats& eff_offset = (rearm_mode || effective_suspected) ? recent : full;
    const SliceStats& eff_flip   = (rearm_mode || effective_suspected) ? recent : full;
    bool full_offset_ok = ballistic_rearm ? (eff_offset.max_offset <= max_line_offset_px * ballistic_rearm_offset_ratio_) : (eff_offset.max_offset <= max_line_offset_px * trigger_max_full_offset_ratio);
    bool flip_ok = ballistic_rearm ? (eff_flip.sign_flip_ratio <= ballistic_rearm_max_sign_flip_ratio_) : (eff_flip.sign_flip_ratio <= trigger_max_sign_flip_ratio);
    int min_valid_steps = effective_suspected ? std::max(1, trigger_min_recent_valid_steps_ - 1) : trigger_min_recent_valid_steps_;
    double min_valid_ratio = effective_suspected ? std::max(0.40, trigger_min_recent_valid_ratio_ - 0.15) : trigger_min_recent_valid_ratio_;
    bool motion_ok = recent.valid_step_count >= min_valid_steps && recent.valid_step_ratio >= min_valid_ratio;
    bool compact_ok = recent.path_over_net <= trigger_max_path_over_net_;
    double track_age_ms = state.first_seen_ts >= 0 ? (ts - state.first_seen_ts) / 1000.0 : 0.0;
    double disp_per_step = recent.total_disp / std::max(1, recent.valid_step_count);
    std::string birth_kind = state.birth_assign_kind.empty() ? "raw_id" : state.birth_assign_kind;
    bool is_raw_birth = (birth_kind == "raw_id") && !effective_suspected && !rearm_mode;
    bool sparse_burst_veto = is_raw_birth && recent.valid_step_count <= trigger_sparse_burst_max_valid_steps && recent.total_disp >= trigger_sparse_burst_min_total_disp_px && disp_per_step >= trigger_sparse_burst_min_disp_per_step_px && recent.path_over_net <= trigger_sparse_burst_max_path_over_net;
    bool raw_birth_steps_ok = (!is_raw_birth) || recent.valid_step_count >= trigger_raw_min_valid_steps;
    bool old_raw_steps_ok = (!is_raw_birth) || track_age_ms < trigger_old_raw_age_ms || recent.valid_step_count >= trigger_old_raw_min_valid_steps;
    double full_avg_speed = track_age_ms > 0 ? full.total_disp / track_age_ms : 0.0;
    double recent_avg_speed = estimate_recent_speed_px_per_ms(state);
    if (ballistic_rearm && rearm_speed > 0) recent_avg_speed = rearm_speed;
    bool speed_ok = true;
    if (trigger_min_avg_speed_px_per_ms > 0) {
        if (ballistic_rearm || effective_suspected || rearm_mode) speed_ok = recent_avg_speed >= trigger_min_avg_speed_px_per_ms;
        else speed_ok = std::max(full_avg_speed, recent_avg_speed) >= trigger_min_avg_speed_px_per_ms;
    }
    double trigger_disp_thresh = min_total_displacement_px;
    if (recent.direction_ok_ratio >= 0.82 && recent.max_offset <= max_line_offset_px) trigger_disp_thresh = std::min(trigger_disp_thresh, std::max(10.0, min_total_displacement_px * 0.72));
    if (birth_kind == "assoc") trigger_disp_thresh = std::min(trigger_disp_thresh, std::max(10.0, min_total_displacement_px * 0.80));
    if (effective_suspected) trigger_disp_thresh = std::min(trigger_disp_thresh, std::max(8.0, min_total_displacement_px * 0.60));
    if (rearm_mode) trigger_disp_thresh = std::min(trigger_disp_thresh, std::max(10.0, min_total_displacement_px * 0.65));
    bool trigger_ok = !late_trigger_veto && !sparse_burst_veto && raw_birth_steps_ok && old_raw_steps_ok && recent.total_disp >= trigger_disp_thresh && line_ok && direction_ok && motion_ok && compact_ok && full_offset_ok && flip_ok && speed_ok;
    out["keep_now"] = trigger_ok; out["trigger_ok"] = trigger_ok; out["confirmed_now"] = trigger_ok; out["line_ok"] = line_ok; out["direction_ok"] = direction_ok; out["motion_ok"] = motion_ok;
    out["recent_geom_ok"] = line_ok; out["recent_motion_ok"] = motion_ok; out["trigger_disp_thresh"] = trigger_disp_thresh; out["compact_ok"] = compact_ok;
    out["rearm_mode"] = rearm_mode; out["ballistic_rearm"] = ballistic_rearm; out["ballistic_rearm_window"] = rearm_window; out["ballistic_rearm_speed_px_ms"] = rearm_speed;
    out["post_outlier_rearm"] = post_outlier_rearm; out["post_outlier_rearm_window"] = post_outlier_rearm ? rearm_window : 0; out["post_outlier_rearm_dt_us"] = post_outlier_rearm ? post_dt_us : 0;
    out["required_trigger_streak_override"] = post_outlier_rearm ? post_outlier_rearm_required_streak_ : (ballistic_rearm ? ballistic_rearm_required_streak_ : 0);
    out["reject_late_trigger"] = late_trigger_veto; out["reject_sparse_burst"] = sparse_burst_veto; out["reject_raw_steps"] = !raw_birth_steps_ok; out["reject_old_raw_steps"] = !old_raw_steps_ok;
    out["reject_disp"] = recent.total_disp < trigger_disp_thresh; out["reject_speed"] = !speed_ok; out["phase_code"] = trigger_ok ? (post_outlier_rearm ? 4 : (ballistic_rearm ? 3 : (rearm_mode ? 2 : 10))) : -1;
    return out;
}
bool CppLineMotionFilter::size_ok_for_maintain(TrackState& state, const Cluster& cluster) const {
    if (!state.last_cluster.has_value()) return true;
    const auto& lc = *state.last_cluster;
    double ratio = std::max({cluster.width/std::max(1.0,lc.width), lc.width/std::max(1.0,cluster.width), cluster.height/std::max(1.0,lc.height), lc.height/std::max(1.0,cluster.height)});
    return ratio <= maintain_size_ratio_;
}
void CppLineMotionFilter::update_segment_state(TrackState& state, int64_t ts) {
    (void)ts;
    auto dir = track_direction(state);
    if (!dir.has_value()) return;
    if (!state.segment_dir.has_value()) { state.segment_dir = dir; return; }
    double cosv = dot2(state.segment_dir->first, state.segment_dir->second, dir->first, dir->second);
    cosv = clampd(cosv, -1.0, 1.0);
    if (cosv < segment_turn_cos_) {
        state.turn_count += 1;
        state.cumulative_turn_deg += std::acos(cosv) * 180.0 / M_PI;
        state.segment_index += 1;
        state.segment_dir = dir;
    } else {
        // low-pass update keeps direction stable while preserving original segment continuity intent
        double nx = 0.8*state.segment_dir->first + 0.2*dir->first;
        double ny = 0.8*state.segment_dir->second + 0.2*dir->second;
        double n = hypot2(nx, ny);
        if (n > 1e-6) state.segment_dir = std::make_pair(nx/n, ny/n);
    }
}
std::map<std::string,double> CppLineMotionFilter::evaluate_probation_and_maintain(TrackState& state, const Cluster& cluster, int64_t ts) {
    std::map<std::string,double> out;
    auto full_opt = compute_stats(state.points);
    auto recent_opt = slice_stats(last_n_point5(state.points, recent_window_points_));
    SliceStats full = full_opt.value_or(SliceStats{});
    SliceStats recent = recent_opt.value_or(SliceStats{});
    double recent_total_disp = recent.total_disp;
    int recent_valid_steps = recent.valid_step_count;
    double recent_valid_ratio = recent.valid_step_ratio;
    double recent_path_over_net = recent.path_over_net;
    double recent_max_offset = recent.max_offset;
    double recent_dir_ratio = recent.direction_ok_ratio;
    double recent_same_sign = recent.same_sign_ratio;
    double recent_sign_flip = recent.sign_flip_ratio;
    double recent_speed_px_ms = estimate_recent_speed_px_per_ms(state);
    bool recent_linear_fast_ok = recent_valid_steps >= 2 && recent_total_disp >= std::max(12.0, min_total_displacement_px * 0.55) && recent_path_over_net <= 1.35 &&
        recent_max_offset <= std::max(maintain_recent_offset_px * 1.25, max_line_offset_px * 1.20) && recent_dir_ratio >= 0.78 && recent_same_sign >= 0.78 && recent_sign_flip <= 0.30 &&
        recent_speed_px_ms >= std::max(0.45, trigger_min_avg_speed_px_per_ms * 0.75);

    if (model_track_enabled && model_outlier_kill_enabled && state.model_outlier_count >= model_outlier_kill_frames) {
        if (state.impact_state.empty()) {
            state.impact_state = "impact_candidate"; state.impact_start_ts = ts; state.impact_outlier_frames = 1; state.impact_last_ts = ts;
            if (state.impact_x == 0.0 && state.impact_y == 0.0) { state.impact_x = state.model_ready ? state.model_pred_x : cluster.cx; state.impact_y = state.model_ready ? state.model_pred_y : cluster.cy; }
            push_recent_segment(state, ts, "impact_candidate");
            push_occlusion_owner(state, ts, "impact_candidate");
        } else { state.impact_state = "occlusion_predict"; state.impact_outlier_frames += 1; state.impact_last_ts = ts; }
        int64_t impact_age_us = state.impact_start_ts >= 0 ? ts - state.impact_start_ts : 0;
        bool allow_short_hold = state.impact_outlier_frames <= 2 && impact_age_us <= 30000;
        if (allow_short_hold) {
            state.p12_polluted_tail = true;
            out["keep_now"] = 1; out["maintain_ok"] = 1; out["phase_code"] = 20; out["trigger_ok"] = 0; out["confirmed_now"] = 0; out["recent_geom_ok"] = 1; out["recent_motion_ok"] = 1; out["compact_ok"] = 1;
            return out;
        }
        if (state.impact_x != 0.0 || state.impact_y != 0.0) { state.model_x = state.impact_x; state.model_y = state.impact_y; state.model_points.push_back(Point3{state.impact_x,state.impact_y,ts}); }
        state.p12_polluted_tail = true;
        push_occlusion_owner(state, ts, "model_outlier");
        deactivate_bullet(state, ts, "model_outlier");
        out["keep_now"] = 0; out["maintain_ok"] = 0; out["phase_code"] = -20; out["trigger_ok"] = 0; out["confirmed_now"] = 0; out["recent_geom_ok"] = 0; out["recent_motion_ok"] = 0;
        return out;
    }
    if (!state.impact_state.empty() && !state.model_outlier) { state.impact_state.clear(); state.impact_start_ts = -1; state.impact_outlier_frames = 0; state.impact_last_ts = -1; }

    bool recent_motion_ok = recent_total_disp >= recent_static_disp_px_ || recent_valid_steps >= 1 || recent_valid_ratio >= 0.20;
    bool recent_geom_ok = size_ok_for_maintain(state, cluster);
    auto post = post_assign_stats(state, ts);
    double post_disp = mget(post, "total_disp", 0.0); int post_steps = (int)mget(post, "valid_step_count", 0.0);
    bool probation_new_motion_ok = true;
    if (state.birth_assign_kind == "raw_id" && !state.probation_passed) probation_new_motion_ok = post_disp >= probation_new_min_extra_disp_px && post_steps >= probation_new_min_extra_valid_steps;
    bool is_reuse_probation = reuse_probation_birth_kinds_.count(state.birth_assign_kind) && !state.probation_passed;
    bool probation_reuse_motion_ok = true;
    bool probation_reuse_soft_hold = false;
    if (is_reuse_probation) {
        bool strict_ok = post_disp >= std::max(10.0, probation_new_min_extra_disp_px) && post_steps >= std::max(2, probation_new_min_extra_valid_steps) && recent_path_over_net <= 2.20 &&
            recent_max_offset <= std::max(probation_max_offset_px, max_line_offset_px * 1.20) && recent_dir_ratio >= 0.70 && recent_same_sign >= 0.70 && recent_sign_flip <= 0.35;
        double cur_w = std::max(1.0, cluster.width), cur_h = std::max(1.0, cluster.height), area = cur_w * cur_h, long_side = std::max(cur_w, cur_h);
        bool bbox_ok = area <= reuse_probation_max_bbox_area_ && long_side <= reuse_probation_max_bbox_long_side_;
        bool soft_ok = bbox_ok && recent_total_disp >= reuse_probation_min_disp_px_ && recent_valid_steps >= 1 && recent_speed_px_ms >= reuse_probation_min_speed_px_ms_ && recent_path_over_net <= reuse_probation_max_path_over_net_ && recent_max_offset <= reuse_probation_max_offset_px_ && recent_dir_ratio >= reuse_probation_min_dir_ratio_ && recent_same_sign >= reuse_probation_min_same_sign_ && recent_sign_flip <= reuse_probation_max_flip_ratio_;
        probation_reuse_motion_ok = strict_ok || soft_ok;
        int64_t elapsed = state.bullet_assign_ts >= 0 ? ts - state.bullet_assign_ts : 0;
        probation_reuse_soft_hold = !probation_reuse_motion_ok && bbox_ok && elapsed <= reuse_probation_grace_us_ && recent_speed_px_ms >= std::max(0.30, reuse_probation_min_speed_px_ms_ * 0.75) && recent_path_over_net <= reuse_probation_max_path_over_net_ * 1.25 && recent_max_offset <= reuse_probation_max_offset_px_ * 1.20;
    }
    if (state.turn_count > maintain_max_turn_count || state.cumulative_turn_deg > maintain_max_cumulative_turn_deg) {
        deactivate_bullet(state, ts, "overturn"); state.points.clear(); state.pre_confirm_points.clear(); state.last_bullet_id.reset();
        out["keep_now"] = 0; out["maintain_ok"] = 0; out["phase_code"] = -30; return out;
    }
    if (!state.probation_passed) {
        int64_t elapsed = state.bullet_assign_ts >= 0 ? ts - state.bullet_assign_ts : 0;
        bool enough_time = elapsed >= probation_min_elapsed_us_ || ts >= state.probation_until_ts;
        bool continue_disp_ok = true;
        if (state.probation_ref_pos) {
            double dx = (state.model_ready ? state.model_x : cluster.cx) - state.probation_ref_pos->first;
            double dy = (state.model_ready ? state.model_y : cluster.cy) - state.probation_ref_pos->second;
            continue_disp_ok = hypot2(dx,dy) >= probation_continue_disp_px_ || recent_total_disp >= probation_continue_disp_px_;
        }
        bool geom_motion_ok = recent_geom_ok && recent_motion_ok && recent_valid_steps >= probation_recent_valid_steps_ && recent_path_over_net <= probation_max_path_over_net_;
        bool pass_ok = enough_time && geom_motion_ok && continue_disp_ok && probation_new_motion_ok && probation_reuse_motion_ok;
        if (pass_ok) {
            state.probation_passed = true; state.probation_fail_count = 0; if (!state.display_bullet_id) promote_display_id(state);
            out["keep_now"] = 1; out["maintain_ok"] = 1; out["phase_code"] = 30; out["confirmed_now"] = 1; out["recent_geom_ok"] = recent_geom_ok; out["recent_motion_ok"] = recent_motion_ok; out["compact_ok"] = 1; return out;
        }
        if (!recent_geom_ok || !recent_motion_ok || !probation_new_motion_ok || !probation_reuse_motion_ok) state.probation_fail_count += 1;
        if (probation_reuse_soft_hold) state.probation_fail_count = std::max(0, state.probation_fail_count - 1);
        if (state.probation_fail_count > probation_fail_grace_) {
            int old = state.bullet_id.value_or(-1); deactivate_bullet(state, ts, is_reuse_probation ? "probation_reuse_motion" : "probation");
            if (old > 0) displayed_bullet_ids_.erase(old);
            out["keep_now"] = 0; out["maintain_ok"] = 0; out["phase_code"] = -31; out["revoked_bullet_id"] = old; return out;
        }
        out["keep_now"] = 1; out["maintain_ok"] = 1; out["phase_code"] = 31; out["recent_geom_ok"] = recent_geom_ok; out["recent_motion_ok"] = recent_motion_ok; out["compact_ok"] = 1; return out;
    }
    if (recent_total_disp <= recent_static_disp_px_ && recent_valid_steps == 0) state.static_fail_count += 1;
    else if (full_opt && full.total_disp <= long_static_disp_px_ && recent_valid_steps == 0) state.static_fail_count += 1;
    else state.static_fail_count = 0;
    bool maintain_ok = recent_geom_ok && state.static_fail_count < stationary_terminate_frames_;
    if (maintain_ok) { update_segment_state(state, ts); out["keep_now"] = 1; out["maintain_ok"] = 1; out["phase_code"] = 40; }
    else { deactivate_bullet(state, ts, "static"); out["keep_now"] = 0; out["maintain_ok"] = 0; out["phase_code"] = -40; }
    out["trigger_ok"] = 0; out["confirmed_now"] = 0; out["recent_geom_ok"] = recent_geom_ok; out["recent_motion_ok"] = recent_motion_ok; out["compact_ok"] = 1;
    return out;
}
void CppLineMotionFilter::reset_model(TrackState& state) {
    state.model_ready = false; state.model_x = state.model_y = state.model_vx = state.model_vy = 0.0; state.model_last_ts = -1; state.model_residual_px = -1.0; state.model_pred_x = state.model_pred_y = 0.0; state.model_obs_used = false; state.model_outlier = false; state.model_outlier_count = 0; state.model_update_mode = "reset"; state.model_speed_px_ms = 0.0; state.model_points.clear();
}
void CppLineMotionFilter::seed_model_from_obs_history(TrackState& state) {
    auto pts = state.obs_points.to_vector();
    if (pts.size() < 2) {
        auto p5 = state.points.to_vector();
        pts.clear(); for (const auto& p : p5) pts.push_back(Point3{p.x,p.y,p.ts});
    }
    if (pts.size() < 2) return;
    const auto& a = pts[pts.size()-2]; const auto& b = pts.back();
    double dt_ms = std::max(0.001, (double)(b.ts - a.ts)/1000.0);
    double vx = (b.x - a.x)/dt_ms, vy = (b.y - a.y)/dt_ms;
    double speed = hypot2(vx, vy);
    if (speed < model_min_init_speed_px_ms && pts.size() >= 3) {
        const auto& c = pts[pts.size()-3]; dt_ms = std::max(0.001, (double)(b.ts - c.ts)/1000.0); vx=(b.x-c.x)/dt_ms; vy=(b.y-c.y)/dt_ms; speed=hypot2(vx,vy);
    }
    state.model_x=b.x; state.model_y=b.y; state.model_vx=vx; state.model_vy=vy; state.model_last_ts=b.ts; state.model_ready=true; state.model_update_mode="seed"; state.model_speed_px_ms=speed; state.model_points.clear();
    for (const auto& p : pts) state.model_points.push_back(p);
}
std::pair<double,double> CppLineMotionFilter::predict_model_xy(const TrackState& state, int64_t ts) const {
    if (!state.model_ready || state.model_last_ts < 0) return {state.model_x, state.model_y};
    double dt_ms = (double)(ts - state.model_last_ts)/1000.0;
    return {state.model_x + state.model_vx * dt_ms, state.model_y + state.model_vy * dt_ms};
}
void CppLineMotionFilter::update_model_with_observation(TrackState& state, double obs_x, double obs_y, int64_t ts) {
    if (!model_track_enabled) return;
    append_point_if_new(state.obs_points, Point3{obs_x, obs_y, ts});
    if (!state.model_ready) { seed_model_from_obs_history(state); return; }
    auto pred = predict_model_xy(state, ts);
    state.model_pred_x = pred.first; state.model_pred_y = pred.second;
    double residual = hypot2(obs_x - pred.first, obs_y - pred.second);
    state.model_residual_px = residual;
    state.model_outlier = residual > model_outlier_residual_px;
    if (state.model_outlier) { state.model_outlier_count += 1; state.model_obs_used = false; state.model_update_mode = "outlier"; }
    else {
        double dt_ms = std::max(0.001, (double)(ts - state.model_last_ts)/1000.0);
        double gain = residual <= model_soft_residual_px ? model_correction_gain : model_weak_correction_gain;
        double new_x = pred.first + gain*(obs_x - pred.first);
        double new_y = pred.second + gain*(obs_y - pred.second);
        double obs_vx = (new_x - state.model_x)/dt_ms;
        double obs_vy = (new_y - state.model_y)/dt_ms;
        // acceleration guard
        double dvx = obs_vx - state.model_vx, dvy = obs_vy - state.model_vy;
        double accel = hypot2(dvx, dvy)/dt_ms;
        if (accel > model_max_accel_px_ms2) {
            double s = model_max_accel_px_ms2 / std::max(1e-6, accel);
            obs_vx = state.model_vx + dvx*s; obs_vy = state.model_vy + dvy*s;
        }
        state.model_x = new_x; state.model_y = new_y; state.model_vx = obs_vx; state.model_vy = obs_vy; state.model_last_ts = ts;
        state.model_speed_px_ms = hypot2(state.model_vx, state.model_vy); state.model_obs_used = true; state.model_outlier_count = 0; state.model_update_mode = residual <= model_soft_residual_px ? "normal" : "weak";
        append_point_if_new(state.model_points, Point3{state.model_x, state.model_y, ts});
    }
}
std::optional<SliceStats> CppLineMotionFilter::model_recent_stats(const TrackState& state) const {
    std::vector<Point5> pseudo;
    for (const auto& p : state.model_points.raw()) pseudo.push_back(Point5{p.x,p.y,1.0,1.0,p.ts});
    if (pseudo.size() > 6) pseudo.erase(pseudo.begin(), pseudo.end()-6);
    return slice_stats(pseudo);
}
std::optional<std::pair<double,double>> CppLineMotionFilter::model_direction(const TrackState& state) const {
    if (state.model_points.size() < 2) return std::nullopt;
    const auto& a = state.model_points.front(); const auto& b = state.model_points.back();
    double dx=b.x-a.x, dy=b.y-a.y, n=hypot2(dx,dy);
    if (n < 1e-6) return std::nullopt;
    return std::make_pair(dx/n, dy/n);
}
void CppLineMotionFilter::cleanup_missing(int64_t ts) {
    std::vector<int> to_delete;
    for (auto& kv : tracks_) {
        auto& st = kv.second;
        if (st.bullet_active && st.miss_count > terminate_miss_frames_) deactivate_bullet(st, ts, "missing");
        if (!st.bullet_active && st.miss_count > std::max(max_missed_frames, terminate_miss_frames_) + 3) {
            if (st.confirmed_once || st.last_bullet_id.has_value()) push_recent_segment(st, ts, "cleanup_missing");
            to_delete.push_back(kv.first);
        }
    }
    for (int tid : to_delete) tracks_.erase(tid);
    rebuild_raw_map();
}
std::vector<Cluster> CppLineMotionFilter::update(const std::vector<Cluster>& clusters, int64_t ts) {
    last_update_ts_ = ts;
    last_debug_rows_.clear();
    purge_recent_terminated(ts); purge_recent_segments(ts); purge_occlusion_owners(ts);
    auto assignments = assign_clusters(clusters, ts);
    std::set<int> assigned_tracks;
    std::vector<Cluster> accepted;
    for (size_t ci = 0; ci < clusters.size(); ++ci) {
        Cluster c = clusters[ci];
        int tid;
        std::string assign_kind = "assoc";
        auto ait = assignments.find((int)c.raw_id);
        if (ait != assignments.end()) { tid = ait->second; }
        else { TrackState& ns = create_state(); tid = ns.stable_track_id; assign_kind = "new"; }
        TrackState& st = tracks_.at(tid);
        assigned_tracks.insert(tid);
        bool raw_reused = false;
        if (raw_to_track_.count(c.raw_id) && raw_to_track_[c.raw_id] == tid && assign_kind != "new") { assign_kind = "raw_id"; raw_reused = true; }
        st.miss_count = 0;
        st.last_cluster = c;
        st.current_raw_id = c.raw_id;
        if (st.first_seen_ts < 0) st.first_seen_ts = ts;
        st.raw_id_history.push_back(c.raw_id);
        st.recent_widths.push_back(c.width); st.recent_heights.push_back(c.height);
        append_pre_confirm_point(st, c.cx, c.cy, ts);
        st.obs_points.push_back(Point3{c.cx,c.cy,ts});
        st.points.push_back(Point5{c.cx,c.cy,c.width,c.height,ts});
        update_model_with_observation(st, c.cx, c.cy, ts);
        auto full_opt = compute_stats(st.points);
        auto recent_opt = slice_stats(last_n_point5(st.points, recent_window_points_));
        bool just_assigned = false, confirmed_now = false, keep_now = false;
        std::string phase = "none";
        std::string reject_reason = "none";
        std::vector<DebugRow> backfill;
        if (st.bullet_active && st.bullet_id.has_value()) {
            auto ev = evaluate_probation_and_maintain(st, c, ts);
            keep_now = bget(ev,"keep_now"); confirmed_now = bget(ev,"confirmed_now");
            int pc = (int)mget(ev,"phase_code",0);
            if (pc == 20) phase = st.impact_state == "impact_candidate" ? "impact_candidate" : "impact_occlusion";
            else if (pc == -20) { phase = "terminated_model_outlier"; reject_reason = "model_outlier"; }
            else if (pc == 30) phase = "probation_passed";
            else if (pc == 31) phase = "probation";
            else if (pc == -31) { phase = "terminated_probation"; reject_reason = "probation"; }
            else if (pc == 40) phase = "maintain";
            else if (pc == -40) { phase = "terminated_static"; reject_reason = "maintain_static"; }
        } else {
            auto ev = evaluate_trigger(st, c, ts);
            bool trigger_ok = bget(ev,"trigger_ok");
            if (trigger_ok) {
                std::optional<std::map<std::string,double>> reuse = std::nullopt;
                std::string reuse_source = "new";
                bool near_owner = is_near_occlusion_owner(st, ts);

                // P15: 新 raw_id 已经满足 trigger 时，也必须先尝试继承旧 bullet_id。
                // 旧版只在 trigger 失败分支才走 segment_pool/owner，导致新段一旦很快自证为子弹，
                // 就直接 next_bullet_id_++，同一颗子弹被拆成多个 bullet_id。
                auto own = try_occlusion_owner_capture_new_track(st, ts);
                if (!own.empty() && mget(own, "ok", 0.0) > 0.5) {
                    reuse = own;
                    reuse_source = st.birth_assign_kind.empty() ? "owner_forward" : st.birth_assign_kind;
                }
                if (!reuse) {
                    reuse = try_segment_pool_capture_new_track(st, ts);
                    if (reuse) reuse_source = st.birth_assign_kind.empty() ? "segment_pool" : st.birth_assign_kind;
                }
                if (!reuse) {
                    reuse = try_confirmed_bullet_id_stitch(st, ts);
                    if (reuse) reuse_source = st.birth_assign_kind.empty() ? "bullet_id_stitch" : st.birth_assign_kind;
                }
                if (!reuse && !near_owner) {
                    reuse = try_reuse_same_track_bullet(st, ts);
                    if (reuse) reuse_source = "same_track";
                }
                if (!reuse && !near_owner) {
                    reuse = try_reuse_recent_terminated_bullet(st, ts);
                    if (reuse) reuse_source = st.birth_assign_kind.empty() ? "recent_terminated" : st.birth_assign_kind;
                }

                if (reuse) {
                    int rb = (int)mget(*reuse,"bullet_id",-1); if (rb > 0) st.bullet_id = rb;
                    int db = (int)mget(*reuse,"display_bullet_id",-1); if (db > 0) st.display_bullet_id = db;
                } else if (near_owner) {
                    // owner 污染窗口内，不能解释成旧弹道续接，就不要马上开新 bullet_id，
                    // 让它多积累 1~2 个点后再判断，避免同一颗子弹在障碍物边缘碎成多个 ID。
                    trigger_ok = false;
                    phase = "p12_owner_residual_guard";
                    reject_reason = "p12_owner_residual_guard";
                } else {
                    st.bullet_id = next_bullet_id_++;
                    reuse_source = "new";
                }

                if (!trigger_ok || !st.bullet_id.has_value()) {
                    keep_now = false;
                    confirmed_now = false;
                } else {
                st.bullet_active = true; st.confirmed_once = true; st.static_fail_count = 0; st.bullet_assign_ts = ts; st.probation_ref_pos = std::make_pair(st.model_ready ? st.model_x : c.cx, st.model_ready ? st.model_y : c.cy);
                st.probation_fail_count = 0; st.probation_passed = false; st.probation_until_ts = ts + probation_window_us_; st.probation_start_point_count = (int)st.points.size(); st.probation_start_ts = ts;
                if (reuse_source != "new") st.birth_assign_kind = reuse_source;
                else st.birth_assign_kind = bget(ev,"post_outlier_rearm") ? "post_outlier_rearm" : (bget(ev,"ballistic_rearm") ? "ballistic_rearm" : (raw_reused ? "raw_id" : assign_kind));
                backfill_display_from_preconfirm(st); seed_model_from_obs_history(st); if (st.model_points.size() >= 2) st.display_points.assign(st.model_points.to_vector());
                auto bf = select_confirmed_backfill_points(st, ts); if (!bf.empty()) { apply_confirmed_backfill_points(st,bf); try_promote_display_early(st, full_opt ? &*full_opt : nullptr, recent_opt ? &*recent_opt : nullptr, ts); backfill = build_confirmed_backfill_debug_rows(st,c.raw_id,tid,c,bf,ts); }
                just_assigned = true; confirmed_now = true; keep_now = true; phase = bget(ev,"post_outlier_rearm") ? "trigger_post_outlier_rearm" : (bget(ev,"ballistic_rearm") ? "trigger_ballistic_rearm" : "trigger");
                st.turn_count = 0; st.cumulative_turn_deg = 0.0; st.recent_offset_exceed_count = 0; st.segment_dir = track_direction(st); st.bbox_reuse_min_disp_px = 0.0;
                }
            } else if (assign_kind == "new" && (int)st.points.size() <= std::max(segment_bounce_candidate_max_points_, bullet_id_bounce_link_candidate_max_points_)) {
                auto near_owner = is_near_occlusion_owner(st, ts);
                auto own = try_occlusion_owner_capture_new_track(st, ts);
                std::optional<std::map<std::string,double>> ghost = (!own.empty() && mget(own, "ok", 0.0) > 0.5) ? std::optional<std::map<std::string,double>>(own) : std::nullopt;
                if (!ghost && !near_owner) ghost = try_segment_pool_capture_new_track(st, ts);
                if (!ghost) ghost = try_confirmed_bullet_id_stitch(st, ts);
                if (!ghost && !near_owner && st.points.size() <= 2) ghost = try_ghost_capture_new_track(st, ts);
                if (ghost) {
                    int gb = (int)mget(*ghost,"bullet_id",-1); if (gb > 0) st.bullet_id = gb; else st.bullet_id = next_bullet_id_++;
                    int gd = (int)mget(*ghost,"display_bullet_id",-1); if (gd > 0) st.display_bullet_id = gd;
                    st.bullet_active = true; st.confirmed_once = true; st.bullet_assign_ts = ts; st.probation_passed = false; st.probation_until_ts = ts + probation_window_us_; st.probation_start_ts = ts; st.probation_start_point_count = (int)st.points.size();
                    if (st.birth_assign_kind.empty()) st.birth_assign_kind = mget(*ghost,"bounce",0) ? "ghost_capture_bounce" : "ghost_capture";
                    backfill_display_from_preconfirm(st); seed_model_from_obs_history(st); if (st.model_points.size() >= 2) st.display_points.assign(st.model_points.to_vector());
                    auto bf = select_confirmed_backfill_points(st, ts); if (!bf.empty()) { apply_confirmed_backfill_points(st,bf); try_promote_display_early(st, full_opt ? &*full_opt : nullptr, recent_opt ? &*recent_opt : nullptr, ts); backfill = build_confirmed_backfill_debug_rows(st,c.raw_id,tid,c,bf,ts); }
                    just_assigned = true; confirmed_now = true; keep_now = true; phase = st.birth_assign_kind; st.keep_streak = std::max(st.keep_streak, bullet_min_output_streak);
                } else if (near_owner) { phase = "p12_owner_residual_guard"; keep_now = false; }
            }
        }
        bool accepted_now = st.bullet_active && st.bullet_id.has_value();
        bool suppress_display = st.p12_polluted_tail || st.model_outlier || !st.impact_state.empty() || phase == "impact_candidate" || phase == "impact_occlusion" || phase == "terminated_model_outlier" || phase == "p12_owner_residual_guard";
        int eff_disp = (st.display_bullet_id && !suppress_display) ? *st.display_bullet_id : -1;
        if (accepted_now) accepted.push_back(c);
        for (auto& r: backfill) last_debug_rows_.push_back(std::move(r));
        DebugRow r;
        r.integer["timestamp"] = ts; r.integer["raw_id"] = c.raw_id; r.integer["stable_track_id"] = tid; r.text["assign_kind"] = assign_kind;
        r.numeric["x"] = c.x; r.numeric["y"] = c.y; r.numeric["width"] = c.width; r.numeric["height"] = c.height; r.numeric["cx"] = c.cx; r.numeric["cy"] = c.cy;
        r.numeric["obs_x"] = c.cx; r.numeric["obs_y"] = c.cy; r.numeric["model_x"] = st.model_ready ? st.model_x : c.cx; r.numeric["model_y"] = st.model_ready ? st.model_y : c.cy; r.numeric["pred_x"] = st.model_ready ? st.model_pred_x : c.cx; r.numeric["pred_y"] = st.model_ready ? st.model_pred_y : c.cy;
        r.numeric["model_residual_px"] = st.model_residual_px; r.integer["model_obs_used"] = st.model_obs_used; r.integer["model_outlier"] = st.model_outlier; r.text["model_update_mode"] = st.model_update_mode; r.numeric["model_speed_px_ms"] = st.model_speed_px_ms;
        r.integer["impact_candidate"] = st.impact_candidate; r.numeric["impact_x"] = st.impact_x; r.numeric["impact_y"] = st.impact_y; r.text["impact_state"] = st.impact_state; r.integer["impact_outlier_frames"] = st.impact_outlier_frames;
        r.integer["n_points"] = (int64_t)st.points.size(); r.integer["model_n_points"] = (int64_t)st.model_points.size(); r.integer["miss_count"] = st.miss_count;
        r.integer["keep_now"] = keep_now; r.integer["confirmed_now"] = confirmed_now; r.integer["accepted_now"] = accepted_now; r.integer["bullet_id"] = st.bullet_id.value_or(-1); r.integer["display_bullet_id"] = eff_disp; r.integer["just_assigned_bullet"] = just_assigned;
        r.text["birth_assign_kind"] = st.birth_assign_kind; r.text["p12_owner_mode"] = st.p12_owner_mode; r.integer["p12_owner_bullet_id"] = st.p12_owner_bullet_id; r.text["phase"] = phase; r.text["reject_reason"] = reject_reason;
        r.text["shot_link_type"] = st.shot_link_type; r.text["shot_link_reason"] = st.shot_link_reason;
        r.integer["shot_link_parent_bullet_id"] = st.shot_link_parent_bullet_id;
        r.integer["shot_link_parent_segment_index"] = st.shot_link_parent_segment_index;
        r.integer["shot_link_parent_stable_track_id"] = st.shot_link_parent_stable_track_id;
        r.numeric["shot_link_parent_x"] = st.shot_link_parent_x; r.numeric["shot_link_parent_y"] = st.shot_link_parent_y;
        r.numeric["shot_link_gap_ms"] = st.shot_link_gap_ms; r.numeric["shot_link_gap_dist_px"] = st.shot_link_gap_dist_px;
        r.numeric["shot_link_angle_deg"] = st.shot_link_angle_deg; r.numeric["shot_link_score"] = st.shot_link_score;
        r.integer["bullet_active"] = st.bullet_active; r.integer["probation_passed"] = st.probation_passed; r.integer["segment_index"] = st.segment_index; r.integer["confirmed_backfill"] = !backfill.empty(); r.integer["recent_segment_pool_size"] = (int64_t)recent_segments_.size();
        put_stats(r, full_opt ? &*full_opt : nullptr, recent_opt ? &*recent_opt : nullptr);
        last_debug_rows_.push_back(std::move(r));
    }
    for (auto& kv : tracks_) if (!assigned_tracks.count(kv.first)) kv.second.miss_count += 1;
    cleanup_missing(ts);
    rebuild_raw_map();
    return accepted;
}
const std::vector<DebugRow>& CppLineMotionFilter::get_last_debug_rows() const { return last_debug_rows_; }
std::vector<Point3> CppLineMotionFilter::filter_display_spikes(const std::vector<Point3>& pts) const {
    if (!draw_spike_filter_enabled_ || pts.size() < 4) return pts;
    std::vector<Point3> out; out.reserve(pts.size()); out.push_back(pts.front());
    for (size_t i=1; i+1<pts.size(); ++i) {
        const auto& a=pts[i-1]; const auto& b=pts[i]; const auto& c=pts[i+1];
        double ax=b.x-a.x, ay=b.y-a.y, cx=c.x-b.x, cy=c.y-b.y;
        double la=hypot2(ax,ay), lc=hypot2(cx,cy);
        bool keep=true;
        if (la>=draw_spike_min_leg_px_ && lc>=draw_spike_min_leg_px_) {
            double vx=c.x-a.x, vy=c.y-a.y, n=hypot2(vx,vy);
            if (n>1e-6) {
                double off=std::fabs((b.x-a.x)*vy - (b.y-a.y)*vx)/n;
                double cosv=dot2(ax/la,ay/la,cx/lc,cy/lc);
                if (off>draw_spike_perp_px_ && cosv<0.35) keep=false;
            }
        }
        if (keep) out.push_back(b);
    }
    out.push_back(pts.back()); return out;
}
std::vector<double> CppLineMotionFilter::polyfit_predict_1d(const std::vector<double>& t, const std::vector<double>& v, const std::vector<double>& tq, int degree) {
    std::vector<double> out; out.reserve(tq.size());
    if (t.empty() || v.empty() || t.size()!=v.size()) { out.assign(tq.size(), 0.0); return out; }
    if (degree <= 1 || t.size() < 3) {
        double mt=0,mv=0; for(size_t i=0;i<t.size();++i){mt+=t[i]; mv+=v[i];} mt/=t.size(); mv/=v.size();
        double num=0,den=0; for(size_t i=0;i<t.size();++i){num+=(t[i]-mt)*(v[i]-mv); den+=(t[i]-mt)*(t[i]-mt);} double b=den>1e-9?num/den:0.0; double a=mv-b*mt;
        for(double x:tq) out.push_back(a+b*x); return out;
    }
    // Small quadratic normal equation solver.
    double s0=t.size(), s1=0,s2=0,s3=0,s4=0, y0=0,y1=0,y2=0;
    for(size_t i=0;i<t.size();++i){double x=t[i], y=v[i], x2=x*x; s1+=x; s2+=x2; s3+=x2*x; s4+=x2*x2; y0+=y; y1+=x*y; y2+=x2*y;}
    double A[3][4]={{s0,s1,s2,y0},{s1,s2,s3,y1},{s2,s3,s4,y2}};
    for(int col=0; col<3; ++col){int piv=col; for(int r=col+1;r<3;++r) if(std::fabs(A[r][col])>std::fabs(A[piv][col])) piv=r; for(int c=col;c<4;++c) std::swap(A[col][c],A[piv][c]); double div=A[col][col]; if(std::fabs(div)<1e-9){ return polyfit_predict_1d(t,v,tq,1); } for(int c=col;c<4;++c) A[col][c]/=div; for(int r=0;r<3;++r){ if(r==col) continue; double f=A[r][col]; for(int c=col;c<4;++c) A[r][c]-=f*A[col][c]; }}
    double a=A[0][3], b=A[1][3], c=A[2][3]; for(double x:tq) out.push_back(a+b*x+c*x*x); return out;
}
std::vector<Point3> CppLineMotionFilter::extend_display_tail_after_terminate(const TrackState& state, const std::vector<Point3>& pts, int64_t now_ts) const {
    if (!draw_extend_tail_after_terminate_ || pts.size()<2 || state.draw_hold_until_ts < now_ts) return pts;
    std::vector<Point3> out=pts;
    const auto& a=pts[pts.size()-2]; const auto& b=pts.back();
    double dx=b.x-a.x, dy=b.y-a.y, n=hypot2(dx,dy); if(n<1e-6) return out;
    dx/=n; dy/=n; double extend=std::min(30.0, std::max(6.0, n*0.45));
    out.push_back(Point3{b.x+dx*extend, b.y+dy*extend, now_ts});
    return out;
}
std::vector<Point3> CppLineMotionFilter::fit_kinematic_display_path(const std::vector<Point3>& pts) const {
    if (!draw_kinematic_fit_enabled_ || (int)pts.size() < draw_kinematic_min_points_) return pts;
    auto clean = filter_display_spikes(pts);
    if ((int)clean.size() < draw_kinematic_min_points_) return clean;
    if (draw_kinematic_force_straight_line_) {
        const auto& a=clean.front(); const auto& b=clean.back(); double dx=b.x-a.x, dy=b.y-a.y, len=hypot2(dx,dy); if(len<1e-6) return clean;
        int samples=std::min(draw_kinematic_max_samples_, std::max((int)clean.size(), (int)std::ceil(len/std::max(1.0, draw_kinematic_sample_step_px_))));
        std::vector<Point3> out; out.reserve(samples);
        for(int i=0;i<samples;++i){double u=samples==1?0.0:(double)i/(samples-1); int64_t ts=(int64_t)std::llround(a.ts + (b.ts-a.ts)*u); out.push_back(Point3{a.x+dx*u,a.y+dy*u,ts});}
        return out;
    }
    std::vector<double> t,x,y,tq;
    int64_t t0=clean.front().ts;
    for(const auto& p: clean){t.push_back((p.ts-t0)/1000.0); x.push_back(p.x); y.push_back(p.y);} 
    int samples=std::min(draw_kinematic_max_samples_, std::max((int)clean.size(), (int)t.size()*2));
    double tmin=t.front(), tmax=t.back(); for(int i=0;i<samples;++i){double u=samples==1?0.0:(double)i/(samples-1); tq.push_back(tmin+(tmax-tmin)*u);} 
    auto xx=polyfit_predict_1d(t,x,tq, clean.size()>= (size_t)draw_kinematic_quad_min_points_ ? 2 : 1); auto yy=polyfit_predict_1d(t,y,tq, clean.size()>= (size_t)draw_kinematic_quad_min_points_ ? 2 : 1);
    std::vector<Point3> out; for(size_t i=0;i<tq.size();++i) out.push_back(Point3{xx[i],yy[i], t0+(int64_t)std::llround(tq[i]*1000.0)}); return out;
}
std::tuple<int,int,int> CppLineMotionFilter::theme_bgr(const std::string& name) {
    if (name == "cyan" || name == "blue") return {255, 220, 80};
    if (name == "purple") return {255, 80, 220};
    if (name == "pink") return {220, 80, 255};
    if (name == "white") return {255, 255, 255};
    return {0, 255, 255};
}
std::tuple<int,int,int> CppLineMotionFilter::scale_color(const std::tuple<int,int,int>& c, double k) {
    return { (int)clampd(std::get<0>(c)*k,0,255), (int)clampd(std::get<1>(c)*k,0,255), (int)clampd(std::get<2>(c)*k,0,255) };
}
std::tuple<int,int,int> CppLineMotionFilter::blend_color(const std::tuple<int,int,int>& a, const std::tuple<int,int,int>& b, double t) {
    t=clampd(t,0.0,1.0); return { (int)std::llround(std::get<0>(a)*(1-t)+std::get<0>(b)*t), (int)std::llround(std::get<1>(a)*(1-t)+std::get<1>(b)*t), (int)std::llround(std::get<2>(a)*(1-t)+std::get<2>(b)*t) };
}
void CppLineMotionFilter::draw_cyber_trail_roi(const std::vector<Point3>& pts) const {
    // Pixel-level drawing is bridged through cpp_line_motion_filter.py so the unchanged
    // original Python/OpenCV cyber-trail renderer consumes the C++ display/model/hold paths.
    // Keeping the renderer unchanged preserves the original visual effect while the heavy
    // detection, association, backfill and trajectory state update run in C++.
    (void)pts;
}
void CppLineMotionFilter::draw(const std::vector<Cluster>& accepted_clusters) const {
    // The Python compatibility layer calls get_draw_paths() and then invokes the original
    // cyber-trail renderer. The accepted cluster list is still preserved for the original
    // debug-box drawing contract.
    (void)accepted_clusters;
}

}  // namespace bullet_cpp
