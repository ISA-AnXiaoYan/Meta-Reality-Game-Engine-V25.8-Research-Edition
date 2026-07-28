from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np

from active_led_marker_worker import ActiveLedMarkerSidecar, MarkerRoiTracker
from marker_blink_decoder import iter_evp, iter_raw, synthetic_events, write_observations
from marker_replay_fixture import partition_events, synthetic_marker_stream, write_evp_fixture
from validate_ir_id_field_run import validate as validate_field_run


def _args(output: Path, **overrides) -> argparse.Namespace:
    values = {
        "output": str(output),
        "camera": "G",
        "camera_serial": "4110035688",
        "run_id": "IRID_TEST",
        "alpha_ms": 20.0,
        "bit_count": 8,
        "checksum": False,
        "tolerance": 0.35,
        "bin_us": 1000,
        "min_events_per_bin": 8,
        "min_segment_events": 20,
        "merge_gap_us": 2500,
        "polarity": "any",
        "roi": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MarkerDecoderTests(unittest.TestCase):
    def test_schema_v2_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            for marker_id in (7, 13):
                out = Path(tmp) / f"id_{marker_id}.jsonl"
                stats = write_observations(
                    [synthetic_events(marker_id, alpha_us=20000, pulse_width_us=10000, checksum=False)],
                    args=_args(out),
                    source="synthetic_self_test",
                )
                rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([marker_id], [int(row["marker_id"]) for row in rows])
                self.assertEqual(2, rows[0]["schema_version"])
                self.assertEqual("ir-id-blink-v1", rows[0]["protocol_version"])
                self.assertEqual(rows[0]["event_ts_us"], rows[0]["ts_us"])
                self.assertTrue(str(rows[0]["decoder_config_hash"]).startswith("sha256:"))
                self.assertEqual(0, stats["extractor_out_of_order_bins"])

    def test_arbitrary_slice_boundaries(self):
        evs = synthetic_events(7, alpha_us=20000, pulse_width_us=10000, checksum=False)
        with tempfile.TemporaryDirectory() as tmp:
            for slice_us in (1000, 4000, 8000, 12000):
                for phase_us in (0, 317, 1000, 3000):
                    if phase_us >= slice_us:
                        continue
                    out = Path(tmp) / f"slice_{slice_us}_{phase_us}.jsonl"
                    write_observations(
                        partition_events(evs, slice_us=slice_us, phase_us=phase_us),
                        args=_args(out),
                        source="replay_fixture",
                    )
                    ids = [int(json.loads(line)["marker_id"]) for line in out.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual([7], ids, (slice_us, phase_us, ids))

    def test_evp_fixture_round_trip(self):
        evs = synthetic_marker_stream([7, 13], repeats=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_evp_fixture(root, evs, slice_us=4000, phase_us=317)
            out = root / "observations.jsonl"
            write_observations(
                iter_evp(manifest["data_file"], manifest=str(root / "event_slices_manifest.json")),
                args=_args(out),
                source="evp_offline",
            )
            ids = [int(json.loads(line)["marker_id"]) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([7, 13], ids)

    def test_raw_adapter_uses_events_iterator_contract(self):
        fake_events = synthetic_events(7, alpha_us=20000, pulse_width_us=10000, checksum=False)
        calls = []

        class FakeIterator:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def __iter__(self):
                yield fake_events[:10]
                yield fake_events[10:20]

        old_core = sys.modules.get("metavision_core")
        old_event_io = sys.modules.get("metavision_core.event_io")
        core = types.ModuleType("metavision_core")
        event_io = types.ModuleType("metavision_core.event_io")
        event_io.EventsIterator = FakeIterator
        core.event_io = event_io
        sys.modules["metavision_core"] = core
        sys.modules["metavision_core.event_io"] = event_io
        try:
            with tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp) / "sample.raw"
                raw.write_bytes(b"fixture")
                batches = list(iter_raw(str(raw), delta_t_us=4000, start_ts_us=123, max_duration_us=9999))
            self.assertEqual(2, len(batches))
            self.assertEqual(4000, calls[0]["delta_t"])
            self.assertEqual(123, calls[0]["start_ts"])
            self.assertEqual(9999, calls[0]["max_duration"])
        finally:
            if old_core is None:
                sys.modules.pop("metavision_core", None)
            else:
                sys.modules["metavision_core"] = old_core
            if old_event_io is None:
                sys.modules.pop("metavision_core.event_io", None)
            else:
                sys.modules["metavision_core.event_io"] = old_event_io


class MarkerSidecarTests(unittest.TestCase):
    def test_roi_tracker(self):
        tracker = MarkerRoiTracker((500, 270, 24, 24), frame_width=1280, frame_height=720, search_margin_px=8)
        initial = tracker.capture_roi()
        tracker.update(540.0, 300.0)
        moved = tracker.capture_roi()
        self.assertNotEqual(initial, moved)
        tracker.reset()
        self.assertEqual(initial, tracker.capture_roi())

    def test_default_off_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = ActiveLedMarkerSidecar(
                enabled=False,
                camera="G",
                camera_serial="4110035688",
                roi="",
                output_path=str(root / "obs.jsonl"),
                status_path=str(root / "status.json"),
            )
            self.assertFalse(sidecar.submit(np.asarray([], dtype=synthetic_events(7, alpha_us=20000, pulse_width_us=10000, checksum=False).dtype), 0, 0))
            sidecar.close()
            self.assertFalse((root / "obs.jsonl").exists())
            self.assertFalse((root / "status.json").exists())

    def test_async_sidecar_and_stale(self):
        evs = synthetic_events(7, alpha_us=20000, pulse_width_us=10000, checksum=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = ActiveLedMarkerSidecar(
                enabled=True,
                camera="G",
                camera_serial="4110035688",
                roi="496,268,32,32",
                output_path=str(root / "obs.jsonl"),
                status_path=str(root / "status.json"),
                perf_csv_path=str(root / "perf.csv"),
                run_id="IRID_TEST",
                frame_width=1280,
                frame_height=720,
                queue_size=512,
                search_margin_px=8,
                stale_after_ms=50,
                status_interval_ms=20,
                perf_interval_ms=20,
            )
            try:
                batches = list(partition_events(evs, slice_us=4000, phase_us=317))
                first_idx = int(np.floor((int(evs["t"].min()) - 317) / 4000))
                for i, batch in enumerate(batches):
                    sidecar.submit(
                        batch,
                        317 + (first_idx + i + 1) * 4000,
                        int(time.time() * 1_000_000),
                        {"event_local_sync_index": i, "mapped_mvs_sync_index": i - 1},
                    )
                sidecar.submit(
                    np.asarray([], dtype=evs.dtype),
                    int(evs["t"].max()) + 10000,
                    int(time.time() * 1_000_000),
                    {"event_local_sync_index": len(batches), "mapped_mvs_sync_index": len(batches) - 1},
                )
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    if sidecar.status().get("observations_written", 0) >= 1:
                        break
                    time.sleep(0.01)
                self.assertEqual(1, sidecar.status().get("observations_written", 0))
                time.sleep(0.08)
                self.assertTrue(sidecar.status()["stale"])
            finally:
                sidecar.close()
            rows = [json.loads(line) for line in (root / "obs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([7], [int(row["marker_id"]) for row in rows])
            self.assertEqual("event_worker_tap", rows[0]["source"])
            self.assertEqual(2, rows[0]["schema_version"])
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(0, int(status.get("queue_dropped", 0)))
            self.assertFalse(status["worker_alive"])
            self.assertTrue((root / "perf.csv").exists())


class EventWorkerIntegrationContractTests(unittest.TestCase):
    def test_default_off_single_tap_and_no_hal_broker_marker_logic(self):
        worker_name = "metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py"
        project_root = next(
            (parent for parent in Path(__file__).resolve().parents if (parent / worker_name).exists()),
            None,
        )
        if project_root is None:
            self.skipTest("source-tree-only event-worker integration contract")
        event_worker = (project_root / worker_name).read_text(encoding="utf-8")
        broker = (project_root / "native_event_bridge" / "hal_event_fifo_broker.cpp").read_text(
            encoding="utf-8"
        )
        sidecar = (Path(__file__).resolve().parent / "active_led_marker_worker.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("action='store_true', default=False", event_worker)
        self.assertEqual(1, event_worker.count("marker_sidecar.submit("))
        self.assertLess(
            event_worker.index("raw_recorder.submit("),
            event_worker.index("marker_sidecar.submit("),
        )
        self.assertNotIn("active_led_marker", broker)
        self.assertNotIn("HalEventFifoIterator", sidecar)


class LauncherAndFieldValidationContractTests(unittest.TestCase):
    def test_launcher_requires_one_explicit_marker_camera(self):
        root = Path(__file__).resolve().parent
        launcher = (root / "launch_fusion_system.py").read_text(encoding="utf-8")
        self.assertIn("--active-led-marker-cameras", launcher)
        self.assertIn("requires exactly one --active-led-marker-cameras alias", launcher)
        self.assertIn("shared['active_led_marker_enable'] = False", launcher)
        self.assertIn("cam_cfg['active_led_marker_enable'] = bool", launcher)
        self.assertIn("--active-led-marker-queue-size', type=int, default=512", launcher)
        sidecar = (root / "active_led_marker_worker.py").read_text(encoding="utf-8")
        self.assertIn("queue_size: int = 512", sidecar)

    def test_field_validator_accepts_complete_shadow_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observation = root / "observation.jsonl"
            status = root / "status.json"
            perf = root / "perf.csv"
            row = {
                "schema_version": 2,
                "protocol_version": "ir-id-blink-v1",
                "marker_id": 7,
                "run_id": "IRID_V26D_TEST",
                "decoder_config_hash": "sha256:test",
                "stale": False,
            }
            observation.write_text("\n".join(json.dumps(row) for _ in range(3)) + "\n", encoding="utf-8")
            status.write_text(json.dumps({"enabled": True, "state": "stopped", "queue_dropped": 0}), encoding="utf-8")
            perf.write_text("process_ms_p95\n0.42\n", encoding="utf-8")
            report = validate_field_run(
                observation_path=observation,
                status_path=status,
                perf_path=perf,
                expected_id=7,
                run_id="IRID_V26D_TEST",
                min_observations=3,
                max_false_ids=0,
                max_queue_dropped=0,
                max_p95_ms=2.0,
            )
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
