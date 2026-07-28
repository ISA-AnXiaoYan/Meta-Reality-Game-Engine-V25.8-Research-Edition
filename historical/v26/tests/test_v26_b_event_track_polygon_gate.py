from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_track_polygon_gate import BusinessIdV1Shadow, BusinessIdV3Shadow, EventTrackPolygonGate


DTYPE = np.dtype([
    ("id", np.int64), ("x", np.float32), ("y", np.float32),
    ("width", np.float32), ("height", np.float32),
])


class FakeMaskProvider:
    def __init__(self, mask):
        self.mask = mask
        self.available = True

    def get_current_mask(self, current_sync_index, slice_ts_us, current_sync_event_ts_us=None):
        if not self.available:
            return None, None, "current_mask_missing"
        return self.mask, {"sync_index": int(current_sync_index), "mask_pixels": int(np.count_nonzero(self.mask)), "used_persons": 1}, "current_sync_match"


def gate_args(**overrides):
    values = dict(
        event_track_polygon_gate_mode="shadow",
        event_track_polygon_gate_cameras="A",
        event_track_polygon_gate_hold_enable=True,
        event_track_polygon_gate_hold_ms=30.0,
        event_track_polygon_gate_hold_max_sync_gap=2,
        event_track_polygon_gate_log_jsonl="",
        event_track_polygon_gate_log_max_fps=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def shadow_args(**overrides):
    values = dict(
        event_business_id_v1_shadow_enable=True,
        event_business_id_v1_shadow_cameras="A",
        event_business_id_v1_confirm_min_observations=3,
        event_business_id_v1_confirm_min_displacement_px=6.0,
        event_business_id_v1_confirm_min_speed_px_ms=0.05,
        event_business_id_v1_assoc_radius_px=18.0,
        event_business_id_v1_max_gap_ms=120.0,
        event_business_id_v1_collision_window_ms=260.0,
        event_business_id_v1_collision_max_gap_px=95.0,
        event_business_id_v1_shadow_log_jsonl="",
        event_business_id_v1_shadow_log_max_fps=0.0,
        event_business_id_v3_mode="shadow",
        event_business_id_v3_cameras="A",
        event_business_id_v3_size_warmup_observations=3,
        event_business_id_v3_max_size_ratio=2.0,
        event_business_id_v3_size_outlier_consecutive=2,
        event_business_id_v3_log_jsonl="",
        event_business_id_v3_log_max_fps=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class EventTrackPolygonGateTest(unittest.TestCase):
    def test_shadow_keeps_official_input_and_filters_shadow_input(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 255
        clusters = np.array([(1, 10, 10, 2, 2), (2, 24, 24, 2, 2)], dtype=DTYPE)
        gate = EventTrackPolygonGate(gate_args(), 32, 32, "A", FakeMaskProvider(mask))

        primary, shadow, diag = gate.apply(clusters, ts=1000, sync_index=7)

        self.assertEqual(len(primary), 2)
        self.assertEqual(len(shadow), 1)
        self.assertEqual(int(shadow[0]["id"]), 2)
        self.assertEqual(diag["source"], "LIVE")
        self.assertEqual(diag["filtered_cluster_count"], 1)

    def test_hold_has_bounded_time_and_sync_gap(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 255
        clusters = np.array([(1, 10, 10, 2, 2)], dtype=DTYPE)
        provider = FakeMaskProvider(mask)
        gate = EventTrackPolygonGate(gate_args(), 32, 32, "A", provider)
        gate.apply(clusters, ts=1000, sync_index=7)
        provider.available = False

        _, shadow, diag = gate.apply(clusters, ts=20_000, sync_index=8)
        self.assertEqual(len(shadow), 0)
        self.assertEqual(diag["source"], "HOLD")

        _, shadow, diag = gate.apply(clusters, ts=21_000, sync_index=10)
        self.assertEqual(len(shadow), 1)
        self.assertEqual(diag["source"], "NONE")
        self.assertEqual(diag["reason"], "hold_sync_gap")

    def test_empty_cluster_probe_reports_polygon_source(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 255
        gate = EventTrackPolygonGate(gate_args(), 32, 32, "A", FakeMaskProvider(mask))
        empty = np.empty((0,), dtype=DTYPE)

        primary, shadow, diag = gate.apply(empty, ts=1000, sync_index=7)

        self.assertEqual(len(primary), 0)
        self.assertEqual(len(shadow), 0)
        self.assertEqual(diag["source"], "LIVE")
        self.assertTrue(diag["mask_available"])

    def test_v1_and_v3_shadow_confirm_and_flag_persistent_size_jump(self):
        args = shadow_args()
        v1 = BusinessIdV1Shadow(args, "A")
        v3 = BusinessIdV3Shadow(args, "A")
        small = np.array([(1, 0, 0, 4, 4)], dtype=DTYPE)
        assignments = []
        for ts, x in ((0, 0), (10_000, 4), (20_000, 8)):
            sample = small.copy()
            sample[0]["x"] = x
            assignments, _ = v1.update(sample, ts, {"source": "LIVE"})
            v3.update(assignments, ts, {"source": "LIVE"})
        self.assertEqual(assignments[0]["status"], "confirmed")

        large = np.array([(1, 12, 0, 16, 16)], dtype=DTYPE)
        assignments, _ = v1.update(large, 30_000, {"source": "LIVE"})
        first = v3.update(assignments, 30_000, {"source": "LIVE"})
        assignments, _ = v1.update(large, 40_000, {"source": "LIVE"})
        second = v3.update(assignments, 40_000, {"source": "LIVE"})
        self.assertFalse(first["assignments"][0]["size_jump"])
        self.assertTrue(second["assignments"][0]["size_jump"])


if __name__ == "__main__":
    unittest.main()
