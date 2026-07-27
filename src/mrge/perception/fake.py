# SPDX-License-Identifier: AGPL-3.0-only
"""Dependency-free FakeBackend for contract and replay tests."""

from mrge.contracts import Frame


class FakeBackend:
    """Return deterministic observations without GPU, camera, or model files."""

    def predict(self, frame: Frame) -> list[dict[str, object]]:
        return [
            {
                "camera_id": frame.camera_id,
                "frame_index": frame.frame_index,
                "class_name": "synthetic_person",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "confidence": 1.0,
            }
        ]
