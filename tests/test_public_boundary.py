# SPDX-License-Identifier: AGPL-3.0-only
from mrge.adapters.synthetic import frames
from mrge.contracts import Frame, HitCandidate
from mrge.engine.pipeline import process


def test_synthetic_pipeline_is_candidate_only():
    result = process(next(frames(1)))
    assert isinstance(result, HitCandidate)
    assert result.authority == "candidate"


def test_frame_contract_is_stable():
    frame = next(frames(1))
    assert isinstance(frame, Frame)
    assert frame.camera_id == "cam-01"
