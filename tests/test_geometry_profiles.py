# SPDX-License-Identifier: AGPL-3.0-only
import math

from mrge.geometry import GeometryProfile, SensorPose, transform_point


def test_foot_point_inward_profile_is_explicit():
    profile = GeometryProfile("inward_four_sensor_footpoint_v1", "arena_v1", "foot_point", "four_sides_inward")
    assert profile.anchor_mode == "foot_point"
    assert profile.sensor_layout == "four_sides_inward"
    point = transform_point(1.0, 0.0, SensorPose("north", 0.0, 10.0, -math.pi / 2))
    assert math.isclose(point[0], 0.0, abs_tol=1e-9)
    assert math.isclose(point[1], 9.0, abs_tol=1e-9)


def test_sensor_center_is_a_separate_profile_assumption():
    profile = GeometryProfile("sensor_center_parallel_v1", "arena_v1", "sensor_center", "parallel_wall")
    assert profile.anchor_mode != "foot_point"
