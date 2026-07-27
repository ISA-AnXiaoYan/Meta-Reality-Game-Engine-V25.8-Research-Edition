# SPDX-License-Identifier: AGPL-3.0-only
"""Profile-driven sensor geometry; no camera index implies orientation."""

from dataclasses import dataclass
import math
from typing import Literal

AnchorMode = Literal["foot_point", "sensor_center", "custom_anchor"]
SensorLayout = Literal["four_sides_inward", "parallel_wall", "custom"]


@dataclass(frozen=True)
class SensorPose:
    sensor_id: str
    x: float
    y: float
    yaw_rad: float


@dataclass(frozen=True)
class GeometryProfile:
    profile_id: str
    world_frame_id: str
    anchor_mode: AnchorMode
    sensor_layout: SensorLayout
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_id or not self.world_frame_id:
            raise ValueError("profile_id and world_frame_id are required")


def transform_point(local_x: float, local_y: float, pose: SensorPose) -> tuple[float, float]:
    """Transform a sensor-local point into the declared world frame."""
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (pose.x + cosine * local_x - sine * local_y, pose.y + sine * local_x + cosine * local_y)
