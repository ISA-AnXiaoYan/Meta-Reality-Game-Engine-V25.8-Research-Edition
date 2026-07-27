# SPDX-License-Identifier: AGPL-3.0-only
from mrge.filters.event import cluster_events, filter_hot_pixels
from mrge.filters.polygon import stale_polygon, validate_polygon
from mrge.preview.sparse_event import build_sparse_preview


def test_event_filters_are_deterministic_and_explain_rejects():
    events = [{"x": 1, "y": 1, "polarity": 1}, {"x": 2, "y": 1, "polarity": -1}, {"x": 9, "y": 9}]
    kept, stats = filter_hot_pixels(events, {(2, 1)})
    assert len(kept) == 2
    assert stats["hot_pixel_reject_count"] == 1
    assert len(cluster_events(kept, radius=1)) == 2


def test_polygon_and_sparse_preview_are_non_authoritative():
    assert validate_polygon([[0, 0], [1, 0], [1, 1]])["gate_state"] == "pass"
    assert stale_polygon(9, 2)["gate_state"] == "fail_open"
    preview = build_sparse_preview([{"x": 3, "y": 4}])
    assert preview["preview_only"] is True
    assert preview["authority_write"] is False
