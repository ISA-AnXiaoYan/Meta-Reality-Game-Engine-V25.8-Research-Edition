#include "cpp_line_motion_filter.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>
#include <cstring>
#include <cstdint>

namespace py = pybind11;
using namespace bullet_cpp;

static py::dict cluster_to_dict(const Cluster& c) {
    py::dict d;
    d["id"] = c.id;
    d["raw_id"] = c.raw_id;
    d["x"] = c.x;
    d["y"] = c.y;
    d["width"] = c.width;
    d["height"] = c.height;
    d["cx"] = c.cx;
    d["cy"] = c.cy;
    return d;
}

static py::dict point3_to_dict(const Point3& p) {
    py::dict d;
    d["x"] = p.x;
    d["y"] = p.y;
    d["ts"] = p.ts;
    return d;
}

static py::tuple point3_to_tuple(const Point3& p) {
    return py::make_tuple(p.x, p.y, p.ts);
}

static py::dict debug_row_to_dict(const DebugRow& r) {
    py::dict d;
    for (const auto& kv : r.numeric) d[py::str(kv.first)] = kv.second;
    for (const auto& kv : r.integer) d[py::str(kv.first)] = kv.second;
    for (const auto& kv : r.text) d[py::str(kv.first)] = kv.second;
    return d;
}

static double get_float_field(py::handle item, const char* name, double fallback=0.0) {
    try { return py::cast<double>(item[py::str(name)]); }
    catch (...) { return fallback; }
}

static int64_t get_int_field(py::handle item, const char* name, int64_t fallback=0) {
    try { return py::cast<int64_t>(item[py::str(name)]); }
    catch (...) { return fallback; }
}

static std::vector<Cluster> clusters_from_python(py::object clusters_obj) {
    std::vector<Cluster> out;
    if (clusters_obj.is_none()) return out;
    py::sequence seq;
    try { seq = py::reinterpret_borrow<py::sequence>(clusters_obj); }
    catch (const py::error_already_set& e) { throw std::runtime_error(std::string("clusters object is not iterable: ") + e.what()); }
    out.reserve((size_t)py::len(seq));
    for (py::handle item : seq) {
        Cluster c;
        int64_t raw_id = get_int_field(item, "id", get_int_field(item, "raw_id", -1));
        c.id = raw_id;
        c.raw_id = raw_id;
        c.x = get_float_field(item, "x", 0.0);
        c.y = get_float_field(item, "y", 0.0);
        c.width = get_float_field(item, "width", 1.0);
        c.height = get_float_field(item, "height", 1.0);
        c.cx = c.x + 0.5 * c.width;
        c.cy = c.y + 0.5 * c.height;
        out.push_back(c);
    }
    return out;
}



static std::vector<Cluster> clusters_from_numpy(py::array arr) {
    // Robust checkpoint-06 implementation: use numpy field views to build a compact C++
    // vector, then release the GIL only for the heavy C++ state-machine update.  This avoids
    // Python-side dict construction while staying ABI-safe for structured dtypes from
    // Metavision/spatter.
    py::buffer_info info = arr.request();
    if (info.ndim != 1) throw std::runtime_error("clusters numpy array must be 1-D structured array");
    auto get_field = [&](const char* name, bool required) -> py::object {
        try { return arr[py::str(name)]; }
        catch (const py::error_already_set&) {
            if (required) throw std::runtime_error(std::string("clusters numpy dtype missing field: ") + name);
            return py::none();
        }
    };
    py::object ids = get_field("id", false);
    if (ids.is_none()) ids = get_field("raw_id", true);
    py::object xs = get_field("x", true);
    py::object ys = get_field("y", true);
    py::object ws = get_field("width", true);
    py::object hs = get_field("height", true);
    const ssize_t n = info.shape[0];
    std::vector<Cluster> out;
    out.reserve(static_cast<size_t>(std::max<ssize_t>(0, n)));
    for (ssize_t i = 0; i < n; ++i) {
        Cluster c;
        c.id = py::cast<int64_t>(ids[py::int_(i)]);
        c.raw_id = c.id;
        c.x = py::cast<double>(xs[py::int_(i)]);
        c.y = py::cast<double>(ys[py::int_(i)]);
        c.width = py::cast<double>(ws[py::int_(i)]);
        c.height = py::cast<double>(hs[py::int_(i)]);
        c.cx = c.x + 0.5 * c.width;
        c.cy = c.y + 0.5 * c.height;
        out.push_back(c);
    }
    return out;
}

static py::list accepted_from_debug_rows(const std::vector<DebugRow>& rows) {
    py::list out;
    for (const auto& r : rows) {
        auto it = r.integer.find("accepted_now");
        if (it == r.integer.end() || it->second == 0) continue;
        py::dict d = debug_row_to_dict(r);
        // Match the Python LinearMotionFilter.update() accepted cluster contract:
        // the accepted object is a cluster dict enriched with bullet/debug fields.
        if (!d.contains("id") && d.contains("raw_id")) d["id"] = d["raw_id"];
        out.append(d);
    }
    return out;
}

static py::list rows_to_pylist(const std::vector<DebugRow>& rows) {
    py::list out;
    for (const auto& r : rows) out.append(debug_row_to_dict(r));
    return out;
}

static py::list pointdeque_to_list(const LimitedDeque<Point3>& dq) {
    py::list pts;
    for (const auto& p : dq.raw()) pts.append(point3_to_tuple(p));
    return pts;
}

static py::list pointvec_to_list(const std::vector<Point3>& v) {
    py::list pts;
    for (const auto& p : v) pts.append(point3_to_tuple(p));
    return pts;
}

static py::list build_draw_paths(const CppLineMotionFilter& f, bool include_inactive=false) {
    py::list paths;
    int64_t cur_ts = f.last_update_ts_;
    for (const auto& kv : f.tracks_) {
        const TrackState& st = kv.second;
        bool active_draw = st.bullet_active && st.bullet_id.has_value() && st.display_bullet_id.has_value() && st.miss_count <= f.draw_hold_missed_frames;
        bool hold_draw = st.hold_display_bullet_id.has_value() && cur_ts >= 0 && cur_ts <= st.draw_hold_until_ts && st.hold_display_points.size() >= 2;
        if (!include_inactive && !active_draw && !hold_draw) continue;

        std::vector<Point3> pts_src;
        int display_seed = -1;
        std::string source = "none";
        if (hold_draw) {
            pts_src = f.extend_display_tail_after_terminate(st, st.hold_display_points.to_vector(), cur_ts);
            display_seed = st.hold_display_bullet_id.value_or(-1);
            source = "hold_display_points";
        } else if (st.model_points.size() >= 2) {
            pts_src = st.model_points.to_vector();
            display_seed = st.display_bullet_id.value_or(st.bullet_id.value_or(st.stable_track_id));
            source = "model_points";
        } else if (st.display_points.size() >= 2) {
            pts_src = st.display_points.to_vector();
            display_seed = st.display_bullet_id.value_or(st.bullet_id.value_or(st.stable_track_id));
            source = "display_points";
        } else {
            for (const auto& p : st.points.raw()) pts_src.push_back(Point3{p.x,p.y,p.ts});
            display_seed = st.bullet_id.value_or(st.stable_track_id);
            source = "points_fallback";
        }
        if (pts_src.size() < 2 && !include_inactive) continue;
        std::vector<Point3> fitted = f.fit_kinematic_display_path(pts_src);
        py::dict d;
        d["stable_track_id"] = st.stable_track_id;
        d["bullet_id"] = st.bullet_id.value_or(-1);
        d["display_bullet_id"] = st.display_bullet_id.value_or(-1);
        d["hold_display_bullet_id"] = st.hold_display_bullet_id.value_or(-1);
        d["display_seed"] = display_seed;
        d["source"] = source;
        d["active_draw"] = active_draw ? 1 : 0;
        d["hold_draw"] = hold_draw ? 1 : 0;
        d["miss_count"] = st.miss_count;
        d["draw_hold_until_ts"] = st.draw_hold_until_ts;
        d["n_points"] = (int)st.points.size();
        d["n_display_points"] = (int)st.display_points.size();
        d["n_model_points"] = (int)st.model_points.size();
        d["n_hold_display_points"] = (int)st.hold_display_points.size();
        d["points"] = pointvec_to_list(fitted);
        d["raw_display_points"] = pointvec_to_list(pts_src);
        paths.append(d);
    }
    return paths;
}

PYBIND11_MODULE(_cpp_bullet_full_port_stage06, m) {
    m.doc() = "Checkpoint-06 pybind bridge for full handwritten C++ LinearMotionFilter port";

    py::class_<TrackConfig>(m, "TrackConfig")
        .def(py::init<>())
        .def_readwrite("history_len", &TrackConfig::history_len)
        .def_readwrite("bootstrap_frames", &TrackConfig::bootstrap_frames)
        .def_readwrite("max_angle_deg", &TrackConfig::max_angle_deg)
        .def_readwrite("max_line_offset_px", &TrackConfig::max_line_offset_px)
        .def_readwrite("min_step_px", &TrackConfig::min_step_px)
        .def_readwrite("min_total_displacement_px", &TrackConfig::min_total_displacement_px)
        .def_readwrite("max_missed_frames", &TrackConfig::max_missed_frames)
        .def_readwrite("same_direction_ratio_thresh", &TrackConfig::same_direction_ratio_thresh)
        .def_readwrite("bullet_min_output_streak", &TrackConfig::bullet_min_output_streak)
        .def_readwrite("min_valid_step_count", &TrackConfig::min_valid_step_count)
        .def_readwrite("min_valid_step_ratio", &TrackConfig::min_valid_step_ratio)
        .def_readwrite("maintain_max_offset_px", &TrackConfig::maintain_max_offset_px)
        .def_readwrite("maintain_max_static_frames", &TrackConfig::maintain_max_static_frames)
        .def_readwrite("maintain_max_bbox_std", &TrackConfig::maintain_max_bbox_std)
        .def_readwrite("probation_max_offset_px", &TrackConfig::probation_max_offset_px)
        .def_readwrite("ghost_capture_enabled", &TrackConfig::ghost_capture_enabled)
        .def_readwrite("maintain_recent_offset_px", &TrackConfig::maintain_recent_offset_px)
        .def_readwrite("maintain_recent_offset_exceed_frames", &TrackConfig::maintain_recent_offset_exceed_frames)
        .def_readwrite("maintain_bbox_ema_alpha", &TrackConfig::maintain_bbox_ema_alpha)
        .def_readwrite("maintain_bbox_ema_drift_ratio", &TrackConfig::maintain_bbox_ema_drift_ratio)
        .def_readwrite("maintain_max_turn_count", &TrackConfig::maintain_max_turn_count)
        .def_readwrite("maintain_max_cumulative_turn_deg", &TrackConfig::maintain_max_cumulative_turn_deg)
        .def_readwrite("ghost_bounce_max_dist_px", &TrackConfig::ghost_bounce_max_dist_px)
        .def_readwrite("ghost_bounce_min_speed_px_per_ms", &TrackConfig::ghost_bounce_min_speed_px_per_ms)
        .def_readwrite("same_track_bbox_reuse_min_disp_px", &TrackConfig::same_track_bbox_reuse_min_disp_px)
        .def_readwrite("trigger_max_full_offset_ratio", &TrackConfig::trigger_max_full_offset_ratio)
        .def_readwrite("trigger_max_sign_flip_ratio", &TrackConfig::trigger_max_sign_flip_ratio)
        .def_readwrite("trigger_min_avg_speed_px_per_ms", &TrackConfig::trigger_min_avg_speed_px_per_ms)
        .def_readwrite("trigger_raw_min_valid_steps", &TrackConfig::trigger_raw_min_valid_steps)
        .def_readwrite("trigger_sparse_burst_max_valid_steps", &TrackConfig::trigger_sparse_burst_max_valid_steps)
        .def_readwrite("trigger_sparse_burst_min_disp_per_step_px", &TrackConfig::trigger_sparse_burst_min_disp_per_step_px)
        .def_readwrite("trigger_sparse_burst_min_total_disp_px", &TrackConfig::trigger_sparse_burst_min_total_disp_px)
        .def_readwrite("trigger_sparse_burst_max_path_over_net", &TrackConfig::trigger_sparse_burst_max_path_over_net)
        .def_readwrite("trigger_old_raw_age_ms", &TrackConfig::trigger_old_raw_age_ms)
        .def_readwrite("trigger_old_raw_min_valid_steps", &TrackConfig::trigger_old_raw_min_valid_steps)
        .def_readwrite("probation_new_min_extra_disp_px", &TrackConfig::probation_new_min_extra_disp_px)
        .def_readwrite("probation_new_min_extra_valid_steps", &TrackConfig::probation_new_min_extra_valid_steps)
        .def_readwrite("model_track_enabled", &TrackConfig::model_track_enabled)
        .def_readwrite("model_soft_residual_px", &TrackConfig::model_soft_residual_px)
        .def_readwrite("model_hard_residual_px", &TrackConfig::model_hard_residual_px)
        .def_readwrite("model_outlier_residual_px", &TrackConfig::model_outlier_residual_px)
        .def_readwrite("model_correction_gain", &TrackConfig::model_correction_gain)
        .def_readwrite("model_weak_correction_gain", &TrackConfig::model_weak_correction_gain)
        .def_readwrite("model_min_init_speed_px_ms", &TrackConfig::model_min_init_speed_px_ms)
        .def_readwrite("model_max_accel_px_ms2", &TrackConfig::model_max_accel_px_ms2)
        .def_readwrite("model_outlier_kill_enabled", &TrackConfig::model_outlier_kill_enabled)
        .def_readwrite("model_outlier_kill_frames", &TrackConfig::model_outlier_kill_frames)
        .def_readwrite("raw_id_sticky_guard_enabled", &TrackConfig::raw_id_sticky_guard_enabled)
        .def_readwrite("raw_id_sticky_guard_min_speed_px_ms", &TrackConfig::raw_id_sticky_guard_min_speed_px_ms)
        .def_readwrite("raw_id_sticky_guard_static_step_px", &TrackConfig::raw_id_sticky_guard_static_step_px)
        .def_readwrite("raw_id_sticky_guard_backward_px", &TrackConfig::raw_id_sticky_guard_backward_px)
        .def_readwrite("raw_id_sticky_guard_residual_px", &TrackConfig::raw_id_sticky_guard_residual_px)
        .def_readwrite("bullet_id_stitch_enabled", &TrackConfig::bullet_id_stitch_enabled)
        .def_readwrite("bullet_id_stitch_window_ms", &TrackConfig::bullet_id_stitch_window_ms)
        .def_readwrite("bullet_id_stitch_corridor_px", &TrackConfig::bullet_id_stitch_corridor_px)
        .def_readwrite("bullet_id_stitch_min_dir_cos", &TrackConfig::bullet_id_stitch_min_dir_cos)
        .def_readwrite("bullet_id_stitch_min_speed_px_ms", &TrackConfig::bullet_id_stitch_min_speed_px_ms)
        .def_readwrite("bullet_id_stitch_max_size_ratio", &TrackConfig::bullet_id_stitch_max_size_ratio)
        .def_readwrite("bullet_id_stitch_predict_gain", &TrackConfig::bullet_id_stitch_predict_gain)
        .def_readwrite("bullet_id_stitch_active_max_dt_ms", &TrackConfig::bullet_id_stitch_active_max_dt_ms)
        .def_readwrite("bullet_id_stitch_reverse_enabled", &TrackConfig::bullet_id_stitch_reverse_enabled)
        .def_readwrite("bullet_id_stitch_reverse_window_ms", &TrackConfig::bullet_id_stitch_reverse_window_ms)
        .def_readwrite("bullet_id_stitch_reverse_corridor_px", &TrackConfig::bullet_id_stitch_reverse_corridor_px)
        .def_readwrite("bullet_id_stitch_reverse_min_dir_cos", &TrackConfig::bullet_id_stitch_reverse_min_dir_cos)
        .def_readwrite("bullet_id_stitch_reverse_min_local_disp_px", &TrackConfig::bullet_id_stitch_reverse_min_local_disp_px)
        .def_readwrite("bullet_id_stitch_reverse_max_size_ratio", &TrackConfig::bullet_id_stitch_reverse_max_size_ratio)
        .def_readwrite("bullet_id_bounce_link_enabled", &TrackConfig::bullet_id_bounce_link_enabled)
        .def_readwrite("bullet_id_bounce_link_window_ms", &TrackConfig::bullet_id_bounce_link_window_ms)
        .def_readwrite("bullet_id_bounce_link_max_gap_px", &TrackConfig::bullet_id_bounce_link_max_gap_px)
        .def_readwrite("bullet_id_bounce_link_adaptive_max_px", &TrackConfig::bullet_id_bounce_link_adaptive_max_px)
        .def_readwrite("bullet_id_bounce_link_min_turn_angle_deg", &TrackConfig::bullet_id_bounce_link_min_turn_angle_deg)
        .def_readwrite("bullet_id_bounce_link_min_speed_px_ms", &TrackConfig::bullet_id_bounce_link_min_speed_px_ms)
        .def_readwrite("bullet_id_bounce_link_max_size_ratio", &TrackConfig::bullet_id_bounce_link_max_size_ratio)
        .def_readwrite("bullet_id_bounce_link_candidate_max_points", &TrackConfig::bullet_id_bounce_link_candidate_max_points);

    py::class_<AssociationConfig>(m, "AssociationConfig")
        .def(py::init<>())
        .def_readwrite("max_distance_px", &AssociationConfig::max_distance_px)
        .def_readwrite("direction_penalty_px", &AssociationConfig::direction_penalty_px)
        .def_readwrite("max_size_ratio", &AssociationConfig::max_size_ratio);

    py::class_<DrawConfig>(m, "DrawConfig")
        .def(py::init<>())
        .def_readwrite("draw_trajectory", &DrawConfig::draw_trajectory)
        .def_readwrite("history_len", &DrawConfig::history_len)
        .def_readwrite("pre_confirm_history_len", &DrawConfig::pre_confirm_history_len)
        .def_readwrite("hold_missed_frames", &DrawConfig::hold_missed_frames)
        .def_readwrite("connect_max_gap_px", &DrawConfig::connect_max_gap_px)
        .def_readwrite("kinematic_fit_enabled", &DrawConfig::kinematic_fit_enabled)
        .def_readwrite("kinematic_force_straight_line", &DrawConfig::kinematic_force_straight_line)
        .def_readwrite("kinematic_residual_px", &DrawConfig::kinematic_residual_px)
        .def_readwrite("kinematic_max_curve_px", &DrawConfig::kinematic_max_curve_px)
        .def_readwrite("kinematic_max_accel_px_per_ms2", &DrawConfig::kinematic_max_accel_px_per_ms2)
        .def_readwrite("kinematic_sample_step_px", &DrawConfig::kinematic_sample_step_px)
        .def_readwrite("draw_after_terminate_hold_ms", &DrawConfig::draw_after_terminate_hold_ms)
        .def_readwrite("extend_tail_after_terminate", &DrawConfig::extend_tail_after_terminate);

    py::class_<Cluster>(m, "Cluster")
        .def(py::init<>())
        .def_readwrite("id", &Cluster::id)
        .def_readwrite("raw_id", &Cluster::raw_id)
        .def_readwrite("x", &Cluster::x)
        .def_readwrite("y", &Cluster::y)
        .def_readwrite("width", &Cluster::width)
        .def_readwrite("height", &Cluster::height)
        .def_readwrite("cx", &Cluster::cx)
        .def_readwrite("cy", &Cluster::cy)
        .def("to_dict", [](const Cluster& c){ return cluster_to_dict(c); });

    py::class_<CppLineMotionFilter>(m, "CppLineMotionFilter")
        .def(py::init<const TrackConfig&, const AssociationConfig&, const DrawConfig&>(),
             py::arg("track") = TrackConfig(), py::arg("assoc") = AssociationConfig(), py::arg("draw") = DrawConfig())
        .def_static("cluster_to_struct", &CppLineMotionFilter::cluster_to_struct)
        .def("update_structs", &CppLineMotionFilter::update)
        .def("update", [](CppLineMotionFilter& self, py::object clusters, int64_t ts) {
            auto vec = clusters_from_python(clusters);
            {
                py::gil_scoped_release release;
                self.update(vec, ts);
            }
            return accepted_from_debug_rows(self.get_last_debug_rows());
        }, py::arg("clusters"), py::arg("ts"))
        .def("update_numpy", [](CppLineMotionFilter& self, py::array clusters, int64_t ts) {
            auto vec = clusters_from_numpy(clusters);
            {
                py::gil_scoped_release release;
                self.update(vec, ts);
            }
            return accepted_from_debug_rows(self.get_last_debug_rows());
        }, py::arg("clusters"), py::arg("ts"))
        .def("get_last_debug_rows", [](const CppLineMotionFilter& self) {
            return rows_to_pylist(self.get_last_debug_rows());
        })
        .def("get_draw_paths", [](const CppLineMotionFilter& self, bool include_inactive) {
            return build_draw_paths(self, include_inactive);
        }, py::arg("include_inactive") = false)
        .def("debug_state_counts", [](const CppLineMotionFilter& self) {
            py::dict d;
            d["tracks"] = (int)self.tracks_.size();
            d["recent_terminated"] = (int)self.recent_terminated_.size();
            d["recent_segments"] = (int)self.recent_segments_.size();
            d["occlusion_owners"] = (int)self.occlusion_owners_.size();
            d["last_debug_rows"] = (int)self.get_last_debug_rows().size();
            return d;
        });
}
