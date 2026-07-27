# SPDX-License-Identifier: AGPL-3.0-only
from replay.jsonl import read, write


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [{"frame_index": 0, "authority": "candidate"}]
    write(path, rows)
    assert list(read(path)) == rows
