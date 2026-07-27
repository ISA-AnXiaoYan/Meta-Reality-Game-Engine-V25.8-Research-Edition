# SPDX-License-Identifier: AGPL-3.0-only
from replay.jsonl import read, write
from replay.faults import inject


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [{"frame_index": 0, "authority": "candidate"}]
    write(path, rows)
    assert list(read(path)) == rows


def test_fault_injection_covers_replay_failure_modes():
    rows = [{"frame": {"frame_index": index, "payload": {}}} for index in range(3)]
    assert len(inject(rows, ["missing_frame"])) == 2
    assert len(inject(rows, ["duplicate_frame"])) == 4
    assert inject(rows, ["out_of_order"])[0]["frame"]["frame_index"] == 1
    assert inject(rows, ["stale_polygon"])[0]["diagnostics"]["polygon_age_frames"] == 999
    assert inject(rows, ["empty_event"])[0]["frame"]["payload"]["event_pixels"] == []
