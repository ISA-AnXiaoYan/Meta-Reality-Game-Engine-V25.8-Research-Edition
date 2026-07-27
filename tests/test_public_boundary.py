# SPDX-License-Identifier: AGPL-3.0-only
from mrge.adapters.synthetic import frames
from mrge.contracts import Frame, HitCandidate
from mrge.engine.pipeline import process, run
from mrge.perception.fake import FakeBackend
from mrge.adapters import NullAdapter


def test_synthetic_pipeline_is_candidate_only():
    result = process(next(frames(1)))
    assert isinstance(result, HitCandidate)
    assert result.authority == "candidate"


def test_frame_contract_is_stable():
    frame = next(frames(1))
    assert isinstance(frame, Frame)
    assert frame.camera_id == "cam-01"


def test_synthetic_adapter_supports_two_and_eight_camera_shapes():
    assert len(list(frames(1, 2))) == 2
    assert len(list(frames(1, 8))) == 8


def test_fake_backend_is_deterministic_and_local():
    frame = next(frames(1))
    first = FakeBackend().predict(frame)
    second = FakeBackend().predict(frame)
    assert first == second
    assert first[0]["class_name"] == "synthetic_person"


def test_every_input_has_terminal_research_state():
    envelopes = run(frames(2))
    assert len(envelopes) == 2
    assert all(item.terminal_state == "candidate_emitted" for item in envelopes)
    assert all(item.official_result is False for item in envelopes)


def test_null_adapter_does_not_open_hardware():
    assert list(NullAdapter().frames()) == []
