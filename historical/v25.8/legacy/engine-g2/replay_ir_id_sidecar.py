from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from active_led_marker_worker import ActiveLedMarkerSidecar
from marker_blink_decoder import iter_raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a RAW file through the Event-worker IR-ID sidecar.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--camera", default="G")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--perf", required=True)
    parser.add_argument("--raw-delta-t-us", type=int, default=4000)
    parser.add_argument("--queue-size", type=int, default=512)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--alpha-ms", type=float, default=20.0)
    parser.add_argument("--bit-count", type=int, default=8)
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--bin-us", type=int, default=1000)
    parser.add_argument("--min-events-per-bin", type=int, default=8)
    parser.add_argument("--min-segment-events", type=int, default=20)
    parser.add_argument("--merge-gap-us", type=int, default=2500)
    parser.add_argument("--polarity", choices=["any", "positive", "negative", "pos", "neg"], default="any")
    args = parser.parse_args(argv)

    sidecar = ActiveLedMarkerSidecar(
        enabled=True,
        camera=str(args.camera),
        camera_serial=str(args.camera_serial),
        roi=str(args.roi),
        output_path=str(args.output),
        status_path=str(args.status),
        perf_csv_path=str(args.perf),
        run_id=str(args.run_id),
        frame_width=1280,
        frame_height=720,
        queue_size=int(args.queue_size),
        alpha_ms=float(args.alpha_ms),
        bit_count=int(args.bit_count),
        checksum=bool(args.checksum),
        tolerance=float(args.tolerance),
        bin_us=int(args.bin_us),
        min_events_per_bin=int(args.min_events_per_bin),
        min_segment_events=int(args.min_segment_events),
        merge_gap_us=int(args.merge_gap_us),
        polarity=str(args.polarity),
    )
    first_event_ts = None
    replay_start = time.monotonic()
    slices = 0
    source_events = 0
    try:
        for evs in iter_raw(str(args.raw), delta_t_us=int(args.raw_delta_t_us)):
            arr = np.asarray(evs)
            if len(arr):
                ts_us = int(np.max(arr["t"]))
                if first_event_ts is None:
                    first_event_ts = ts_us
            else:
                ts_us = int((first_event_ts or 0) + slices * int(args.raw_delta_t_us))
            if bool(args.realtime) and first_event_ts is not None:
                target_elapsed = max(0.0, (ts_us - first_event_ts) / 1_000_000.0)
                remaining = target_elapsed - (time.monotonic() - replay_start)
                if remaining > 0:
                    time.sleep(remaining)
            sidecar.submit(arr, ts_us, int(time.time() * 1_000_000), {"event_local_sync_index": slices})
            slices += 1
            source_events += int(len(arr))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and int(sidecar.status().get("queue_depth", 0)) > 0:
            time.sleep(0.01)
    finally:
        sidecar.close()
    result = {
        "schema_version": 1,
        "kind": "ir_id_sidecar_raw_replay",
        "raw": str(args.raw),
        "run_id": str(args.run_id),
        "source_slices": int(slices),
        "source_events": int(source_events),
        "status": sidecar.status(),
    }
    result_path = Path(str(args.status) + ".replay.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
