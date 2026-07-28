from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from marker_blink_decoder import EventPulseExtractor, iter_raw, parse_roi


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect extracted marker pulse intervals from a RAW ROI.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-delta-t-us", type=int, default=4000)
    parser.add_argument("--bin-us", type=int, default=1000)
    parser.add_argument("--min-events-per-bin", type=int, default=8)
    parser.add_argument("--min-segment-events", type=int, default=20)
    parser.add_argument("--merge-gap-us", type=int, default=2500)
    parser.add_argument("--polarity", choices=["any", "positive", "negative", "pos", "neg"], default="any")
    parser.add_argument("--bucket-us", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)

    roi = parse_roi(str(args.roi))
    if roi is None:
        raise SystemExit("--roi must be x,y,w,h")
    extractor = EventPulseExtractor(
        bin_us=int(args.bin_us),
        min_events_per_bin=int(args.min_events_per_bin),
        min_segment_events=int(args.min_segment_events),
        merge_gap_us=int(args.merge_gap_us),
        polarity=str(args.polarity),
        roi=roi,
    )
    pulses = []
    fallback_ts = 0
    for evs in iter_raw(str(args.raw), delta_t_us=int(args.raw_delta_t_us)):
        if len(evs):
            fallback_ts = int(np.max(evs["t"]))
        else:
            fallback_ts += int(args.raw_delta_t_us)
        pulses.extend(extractor.process(evs, current_time_us=fallback_ts))
    pulses.extend(extractor.flush())
    intervals = [int(pulses[index + 1].ts_us) - int(pulses[index].ts_us) for index in range(max(0, len(pulses) - 1))]
    bucket = max(1, int(args.bucket_us))
    buckets = Counter(int(round(value / float(bucket)) * bucket) for value in intervals if value > 0)
    report = {
        "schema_version": 1,
        "kind": "ir_id_raw_marker_pulse_probe",
        "raw": str(args.raw),
        "roi": list(roi),
        "pulse_count": len(pulses),
        "interval_count": len(intervals),
        "interval_min_us": min(intervals) if intervals else None,
        "interval_max_us": max(intervals) if intervals else None,
        "interval_p50_us": None if not intervals else float(np.percentile(intervals, 50)),
        "interval_p95_us": None if not intervals else float(np.percentile(intervals, 95)),
        "top_interval_buckets": [
            {"interval_us": int(value), "count": int(count)}
            for value, count in buckets.most_common(max(1, int(args.top_n)))
        ],
        "first_pulses": [
            {"ts_us": int(pulse.ts_us), "x": round(float(pulse.x), 3), "y": round(float(pulse.y), 3),
             "event_count": int(pulse.event_count), "bin_count": int(pulse.bin_count)}
            for pulse in pulses[: max(1, int(args.top_n))]
        ],
        "extractor_out_of_order_bins": int(extractor.out_of_order_bins),
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
