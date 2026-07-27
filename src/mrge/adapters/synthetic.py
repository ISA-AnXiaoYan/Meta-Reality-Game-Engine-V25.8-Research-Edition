# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic synthetic frame source used by tests and demos."""

from mrge.contracts import Frame


def frames(count: int = 3, camera_count: int = 1):
    if camera_count < 1:
        raise ValueError("camera_count must be >= 1")
    for index in range(count):
        for camera_index in range(camera_count):
            yield Frame(
                "synthetic-run",
                "synthetic-epoch",
                f"cam-{camera_index + 1:02d}",
                index * camera_count + camera_index,
                index * 1_000_000,
                {"objects": []},
            )
