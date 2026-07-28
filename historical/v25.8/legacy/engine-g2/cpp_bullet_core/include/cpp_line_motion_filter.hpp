#pragma once

#include "cpp_bullet_types.hpp"
#include <tuple>

namespace bullet_cpp {

class CppLineMotionFilter {
public:
    CppLineMotionFilter(const TrackConfig& track = TrackConfig(), const AssociationConfig& assoc = AssociationConfig(), const DrawConfig& draw = DrawConfig());

    // Python: _cluster_to_dict lines 639-649
    static Cluster cluster_to_struct(int64_t raw_id, double x, double y, double w, double h);
    // Python: _create_state lines 651-661
    TrackState& create_state();
    // Python: _append_point_if_new lines 664-667
    template <class T> static void append_point_if_new(LimitedDeque<T>& dst, const T& point) { if (!dst.empty() && dst.back() == point) return; dst.push_back(point); }
    // Python: _append_pre_confirm_point lines 669-670
    void append_pre_confirm_point(TrackState& state, double x, double y, int64_t ts);
    // Python: _append_display_point lines 672-673
    void append_display_point(TrackState& state, double x, double y, int64_t ts);
    // Python: _backfill_display_from_preconfirm lines 675-677
    void backfill_display_from_preconfirm(TrackState& state);
    // Python: _select_confirmed_backfill_points lines 683-780
    std::vector<Point5> select_confirmed_backfill_points(TrackState& state, int64_t current_ts);
    // Python: _apply_confirmed_backfill_points lines 782-810
    void apply_confirmed_backfill_points(TrackState& state, const std::vector<Point5>& pts);
    // Python: _build_confirmed_backfill_debug_rows lines 812-891
    std::vector<DebugRow> build_confirmed_backfill_debug_rows(TrackState& state, int64_t raw_id, int stable_track_id, const Cluster& current_cluster, const std::vector<Point5>& pts, int64_t current_ts);
    // Python: _trim_points lines 893-896
    std::vector<Point5> trim_points(const LimitedDeque<Point5>& pts, int n) const;
    // Python: _predict_center lines 898-912
    std::pair<double,double> predict_center(const TrackState& state, int64_t ts) const;
    // Python: _track_direction lines 914-928
    std::optional<std::pair<double,double>> track_direction(const TrackState& state) const;
    // Python: _association_cost lines 930-975
    double association_cost(const TrackState& state, const Cluster& cluster, int64_t ts) const;
    // P13: raw_id 粘背景保护
    bool direct_raw_id_association_ok(const TrackState& state, const Cluster& cluster, int64_t ts) const;
    // Python: _assign_clusters lines 977-1025
    std::map<int,int> assign_clusters(const std::vector<Cluster>& clusters, int64_t ts);
    // Python: _rebuild_raw_map lines 1027-1034
    void rebuild_raw_map();
    // Python: _purge_recent_terminated lines 1036-1040
    void purge_recent_terminated(int64_t ts);
    // Python: _purge_recent_segments lines 1045-1049
    void purge_recent_segments(int64_t ts);
    // Python: _purge_occlusion_owners lines 1054-1059
    void purge_occlusion_owners(int64_t ts);
    // Python: _is_confirmed_owner_source lines 1061-1070
    bool is_confirmed_owner_source(const TrackState& state) const;
    // Python: _push_occlusion_owner lines 1072-1134
    void push_occlusion_owner(const TrackState& state, int64_t ts, const std::string& reason);
    // Python: _candidate_motion_summary_for_owner lines 1136-1193
    std::map<std::string,double> candidate_motion_summary_for_owner(const TrackState& state) const;
    // Python: _is_near_occlusion_owner lines 1195-1210
    bool is_near_occlusion_owner(const TrackState& state, int64_t ts) const;
    // Python: _try_occlusion_owner_capture_new_track lines 1212-1332
    std::map<std::string,double> try_occlusion_owner_capture_new_track(TrackState& state, int64_t ts);
    // Python: _push_recent_segment lines 1334-1377
    void push_recent_segment(TrackState& state, int64_t ts, const std::string& reason);
    // Python: _push_recent_terminated lines 1379-1396
    void push_recent_terminated(const TrackState& state, int64_t ts, const std::string& reason);
    // Python: _deactivate_bullet lines 1398-1455
    void deactivate_bullet(TrackState& state, int64_t ts, const std::string& reason);
    // Python: _try_reuse_same_track_bullet lines 1457-1483
    std::optional<std::map<std::string,double>> try_reuse_same_track_bullet(TrackState& state, int64_t ts);
    // Python: _try_reuse_recent_terminated_bullet lines 1485-1529
    std::optional<std::map<std::string,double>> try_reuse_recent_terminated_bullet(TrackState& state, int64_t ts);
    // Python: _try_segment_pool_capture_new_track lines 1531-1630
    std::optional<std::map<std::string,double>> try_segment_pool_capture_new_track(TrackState& state, int64_t ts);
    // P16: confirmed-bullet display/bullet ID stitching after raw_id split.
    std::optional<std::map<std::string,double>> try_confirmed_bullet_id_stitch(TrackState& state, int64_t ts);
    // Python: _try_ghost_capture_new_track lines 1632-1730
    std::optional<std::map<std::string,double>> try_ghost_capture_new_track(TrackState& state, int64_t ts);
    // Python: _is_suspected_continuation lines 1735-1771
    bool is_suspected_continuation(const TrackState& state, int64_t ts) const;
    // Python: _slice_stats lines 1773-1855
    std::optional<SliceStats> slice_stats(const std::vector<Point5>& pts) const;
    // Python: _compute_stats lines 1857-1866
    std::optional<SliceStats> compute_stats(const LimitedDeque<Point5>& pts) const;
    // Python: _estimate_speed_from_points lines 1868-1878
    double estimate_speed_from_points(const std::vector<Point5>& pts) const;
    // Python: _estimate_recent_speed_px_per_ms lines 1880-1887
    double estimate_recent_speed_px_per_ms(const TrackState& state) const;
    // Python: _promote_display_id lines 1889-1893
    int promote_display_id(TrackState& state);
    // Python: _try_promote_display_early lines 1898-1978
    bool try_promote_display_early(TrackState& state, const SliceStats* full_stats, const SliceStats* recent_stats, int64_t ts);
    // Python: _post_assign_stats lines 1980-1992
    std::map<std::string,double> post_assign_stats(TrackState& state, int64_t ts);
    // Python: _required_trigger_streak lines 1994-2008
    int required_trigger_streak(const TrackState& state, bool ballistic_rearm) const;
    // Python: _select_post_outlier_rearm_suffix_stats lines 2014-2046
    std::optional<SliceStats> select_post_outlier_rearm_suffix_stats(const TrackState& state, int64_t ts) const;
    // Python: _reset_points_to_recent_suffix_for_rearm lines 2048-2081
    void reset_points_to_recent_suffix_for_rearm(TrackState& state, int64_t ts);
    // Python: _select_ballistic_rearm_suffix_stats lines 2084-2186
    std::optional<SliceStats> select_ballistic_rearm_suffix_stats(const TrackState& state, int64_t ts) const;
    // Python: _evaluate_trigger lines 2188-2411
    std::map<std::string,double> evaluate_trigger(TrackState& state, const Cluster& cluster, int64_t ts);
    // Python: _size_ok_for_maintain lines 2413-2422
    bool size_ok_for_maintain(TrackState& state, const Cluster& cluster) const;
    // Python: _update_segment_state lines 2424-2464
    void update_segment_state(TrackState& state, int64_t ts);
    // Python: _evaluate_probation_and_maintain lines 2466-2940
    std::map<std::string,double> evaluate_probation_and_maintain(TrackState& state, const Cluster& cluster, int64_t ts);
    // Python: _reset_model lines 2947-2968
    void reset_model(TrackState& state);
    // Python: _seed_model_from_obs_history lines 2970-3033
    void seed_model_from_obs_history(TrackState& state);
    // Python: _predict_model_xy lines 3035-3043
    std::pair<double,double> predict_model_xy(const TrackState& state, int64_t ts) const;
    // Python: _update_model_with_observation lines 3045-3118
    void update_model_with_observation(TrackState& state, double obs_x, double obs_y, int64_t ts);
    // Python: _model_recent_stats lines 3120-3127
    std::optional<SliceStats> model_recent_stats(const TrackState& state) const;
    // Python: _model_direction lines 3129-3141
    std::optional<std::pair<double,double>> model_direction(const TrackState& state) const;
    // Python: _cleanup_missing lines 3143-3161
    void cleanup_missing(int64_t ts);
    // Python: update lines 3167-3602
    std::vector<Cluster> update(const std::vector<Cluster>& clusters, int64_t ts);
    // Python: get_last_debug_rows lines 3604-3605
    const std::vector<DebugRow>& get_last_debug_rows() const;
    // Python: _filter_display_spikes lines 3607-3640
    std::vector<Point3> filter_display_spikes(const std::vector<Point3>& pts) const;
    // Python: _polyfit_predict_1d lines 3643-3650
    static std::vector<double> polyfit_predict_1d(const std::vector<double>& t, const std::vector<double>& v, const std::vector<double>& tq, int degree);
    // Python: _extend_display_tail_after_terminate lines 3652-3688
    std::vector<Point3> extend_display_tail_after_terminate(const TrackState& state, const std::vector<Point3>& pts, int64_t now_ts) const;
    // Python: _fit_kinematic_display_path lines 3690-3800
    std::vector<Point3> fit_kinematic_display_path(const std::vector<Point3>& pts) const;
    // Python: _theme_bgr lines 3803-3810
    static std::tuple<int,int,int> theme_bgr(const std::string& name);
    // Python: _scale_color lines 3813-3815
    static std::tuple<int,int,int> scale_color(const std::tuple<int,int,int>& c, double k);
    // Python: _blend_color lines 3818-3820
    static std::tuple<int,int,int> blend_color(const std::tuple<int,int,int>& a, const std::tuple<int,int,int>& b, double t);
    // Python: _draw_cyber_trail_roi lines 3822-3959
    void draw_cyber_trail_roi(/* cv::Mat& image */ const std::vector<Point3>& pts) const;
    // Python: draw lines 3961-4037
    void draw(/* cv::Mat& image */ const std::vector<Cluster>& accepted_clusters) const;

private:
    void initialize_derived_parameters();

public:
    TrackConfig tc_;
    AssociationConfig ac_;
    DrawConfig dc_;

    bool draw_trajectory = true;
    double draw_connect_max_gap_px = 72.0;
    int draw_hold_missed_frames = 2;
    int draw_history_len = 40;
    int pre_confirm_history_len = 24;
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

    // Derived and private Python fields from __init__; names intentionally match source comments.
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

    bool bullet_id_stitch_enabled = true;
    double bullet_id_stitch_window_ms = 120.0;
    double bullet_id_stitch_corridor_px = 24.0;
    double bullet_id_stitch_min_dir_cos = 0.82;
    double bullet_id_stitch_min_speed_px_ms = 0.32;
    double bullet_id_stitch_max_size_ratio = 5.5;
    double bullet_id_stitch_predict_gain = 0.55;
    double bullet_id_stitch_active_max_dt_ms = 140.0;
    bool bullet_id_stitch_reverse_enabled = true;
    double bullet_id_stitch_reverse_window_ms = 170.0;
    double bullet_id_stitch_reverse_corridor_px = 26.0;
    double bullet_id_stitch_reverse_min_dir_cos = 0.82;
    double bullet_id_stitch_reverse_min_local_disp_px = 10.0;
    double bullet_id_stitch_reverse_max_size_ratio = 5.8;

    bool bullet_id_bounce_link_enabled_ = true;
    double bullet_id_bounce_link_window_ms_ = 260.0;
    double bullet_id_bounce_link_max_gap_px_ = 95.0;
    double bullet_id_bounce_link_adaptive_max_px_ = 190.0;
    double bullet_id_bounce_link_min_turn_angle_deg_ = 95.0;
    double bullet_id_bounce_link_max_dir_cos_ = -0.087155743;
    double bullet_id_bounce_link_min_speed_px_ms_ = 0.38;
    double bullet_id_bounce_link_max_size_ratio_ = 6.5;
    int bullet_id_bounce_link_candidate_max_points_ = 12;

    std::unordered_map<int, TrackState> tracks_;
    std::unordered_map<int64_t, int> raw_to_track_;
    int next_track_id_ = 1;
    int next_bullet_id_ = 1;
    std::unordered_set<int> displayed_bullet_ids_;
    std::vector<DebugRow> last_debug_rows_;
    std::vector<RecentTerminated> recent_terminated_;
    std::vector<RecentSegment> recent_segments_;
    std::vector<OcclusionOwner> occlusion_owners_;

    // Internal constants copied from Python __init__.
    int64_t p12_owner_window_us_ = 240000;
    int64_t p12_owner_guard_us_ = 220000;
    double p12_forward_corridor_px_ = 28.0;
    double p12_forward_min_project_px_ = 14.0;
    double p12_forward_min_dir_cos_ = 0.62;
    double p12_bounce_birth_px_ = 64.0;
    double p12_bounce_adaptive_px_ = 130.0;
    double p12_bounce_min_leave_px_ = 18.0;
    double p12_bounce_min_speed_px_ms_ = 0.55;
    double p12_residual_guard_px_ = 110.0;
    int p12_commit_min_points_ = 4;
    int p12_commit_min_valid_steps_ = 3;
    double p12_commit_max_path_over_net_ = 1.55;
    double p12_commit_max_offset_ratio_ = 1.65;
    double p12_commit_min_dir_ratio_ = 0.78;
    double p12_commit_min_same_sign_ = 0.78;
    double p12_commit_max_flip_ = 0.30;

    int64_t recent_segment_pool_window_us_ = 220000;
    int64_t segment_bounce_dt_us_ = 200000;
    double segment_bounce_max_dist_px_ = 70.0;
    double segment_bounce_adaptive_max_px_ = 120.0;
    double segment_bounce_min_speed_px_ms_ = 0.35;
    double segment_bounce_max_size_ratio_ = 4.0;
    int segment_bounce_candidate_max_points_ = 6;

    bool confirmed_backfill_enabled_ = true;
    int confirmed_backfill_max_points_ = 16;
    int64_t confirmed_backfill_max_age_us_ = 90000;
    int confirmed_backfill_min_points_ = 4;
    double confirmed_backfill_min_speed_px_ms_ = 0.45;
    double confirmed_backfill_max_path_over_net_ = 1.75;
    double confirmed_backfill_offset_ratio_ = 2.0;
    double confirmed_backfill_min_direction_ok_ = 0.70;
    double confirmed_backfill_min_same_sign_ = 0.70;
    double confirmed_backfill_max_sign_flip_ = 0.35;

    bool post_outlier_rearm_enabled_ = true;
    int64_t post_outlier_rearm_window_us_ = 650000;
    int64_t post_outlier_rearm_min_dt_us_ = 3000;
    int post_outlier_rearm_required_streak_ = 1;
    bool post_outlier_rearm_reset_points_ = true;
    std::set<std::string> post_outlier_rearm_reasons_;

    int recent_window_points_ = 5;
    int slow_window_points_ = 8;
    int64_t late_trigger_max_age_us_ = 150000;
    int trigger_min_recent_valid_steps_ = 2;
    double trigger_min_recent_valid_ratio_ = 0.65;
    double trigger_max_path_over_net_ = 2.20;
    double trigger_max_recent_offset_px_ = 6.0;
    double trigger_min_dir_ratio_ = 0.72;
    double trigger_min_same_sign_ = 0.72;

    int64_t probation_window_us_ = 35000;
    int64_t probation_min_elapsed_us_ = 8000;
    double probation_continue_disp_px_ = 9.0;
    int probation_recent_valid_steps_ = 1;
    double probation_max_path_over_net_ = 2.80;
    int probation_fail_grace_ = 3;

    std::set<std::string> reuse_probation_birth_kinds_;
    int64_t reuse_probation_grace_us_ = 18000;
    double reuse_probation_min_disp_px_ = 6.0;
    double reuse_probation_min_speed_px_ms_ = 0.40;
    double reuse_probation_max_path_over_net_ = 2.80;
    double reuse_probation_max_offset_px_ = 6.0;
    double reuse_probation_min_dir_ratio_ = 0.55;
    double reuse_probation_min_same_sign_ = 0.55;
    double reuse_probation_max_flip_ratio_ = 0.55;
    double reuse_probation_max_bbox_area_ = 14400.0;
    double reuse_probation_max_bbox_long_side_ = 160.0;

    int terminate_miss_frames_ = 4;
    double recent_static_disp_px_ = 0.8;
    double long_static_disp_px_ = 2.1;
    int stationary_terminate_frames_ = 5;
    double maintain_size_ratio_ = 5.4;
    int keep_tail_points_ = 4;
    int64_t same_track_reactivate_window_us_ = 100000;
    int64_t ghost_reactivate_window_us_ = 200000;
    double ghost_base_distance_px_ = 200.0;
    double ghost_speed_px_per_us_ = 0.00055;
    double segment_turn_cos_ = 0.70710678;
    int64_t late_trigger_exempt_time_us_ = 150000;
    double late_trigger_exempt_fwd_px_ = 60.0;
    double late_trigger_exempt_bounce_px_ = 30.0;
    int64_t segment_turn_warmup_us_ = 18000;

    bool ballistic_rearm_enabled_ = true;
    int ballistic_rearm_window_min_ = 4;
    int ballistic_rearm_window_max_ = 6;
    double ballistic_rearm_disp_ratio_ = 0.80;
    int ballistic_rearm_min_valid_steps_ = 3;
    double ballistic_rearm_max_path_over_net_ = 1.45;
    double ballistic_rearm_offset_ratio_ = 1.80;
    double ballistic_rearm_min_direction_ok_ratio_ = 0.70;
    double ballistic_rearm_min_same_sign_ratio_ = 0.70;
    double ballistic_rearm_max_sign_flip_ratio_ = 0.45;
    double ballistic_rearm_max_bbox_area_ = 9600.0;
    double ballistic_rearm_max_bbox_long_side_ = 120.0;
    int ballistic_rearm_required_streak_ = 2;

    int64_t draw_after_terminate_hold_us_ = 0;
    bool draw_extend_tail_after_terminate_ = false;
    bool draw_spike_filter_enabled_ = true;
    double draw_spike_perp_px_ = 12.0;
    double draw_spike_min_leg_px_ = 6.0;
    bool draw_kinematic_fit_enabled_ = true;
    bool draw_kinematic_force_straight_line_ = true;
    int draw_kinematic_min_points_ = 4;
    int draw_kinematic_quad_min_points_ = 7;
    double draw_kinematic_base_residual_px_ = 10.0;
    double draw_kinematic_max_curve_px_ = 0.0;
    double draw_kinematic_max_accel_px_per_ms2_ = 0.0;
    double draw_kinematic_sample_step_px_ = 7.0;
    int draw_kinematic_max_samples_ = 72;
    int64_t last_update_ts_ = -1;
};

}  // namespace bullet_cpp
