# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

from mrge.engine.reports import build_reports
from replay.jsonl import read


def test_reports_are_research_only():
    rows = list(read(Path("examples/quickstart_2cam_synthetic.jsonl")))
    reports = build_reports(rows)
    assert set(reports) == {"sync_report", "polygon_report", "world_state_report", "hit_candidate_report", "bev_static_report"}
    assert reports["sync_report"]["frame_count"] == 2
    assert reports["sync_report"]["official_result"] is False
    assert reports["polygon_report"]["records"][0]["authority"] == "candidate"
    assert reports["hit_candidate_report"]["official_result"] is False
