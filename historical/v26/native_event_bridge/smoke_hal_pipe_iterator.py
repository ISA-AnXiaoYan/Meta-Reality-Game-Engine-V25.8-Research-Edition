#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hal_event_pipe_iterator import HalEventPipeIterator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-bin", default=str(Path(__file__).resolve().parent / "build" / "hal_event_pipe_bridge"))
    ap.add_argument("--serial", required=True)
    ap.add_argument("--slice-us", type=int, default=4000)
    ap.add_argument("--wait-mode", choices=["blocking", "poll"], default="poll")
    ap.add_argument("--poll-sleep-us", type=int, default=100)
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--enable-trigger-in", action="store_true")
    ap.add_argument("--enable-erc", action="store_true")
    ap.add_argument("--erc-rate", type=int, default=10000000)
    ap.add_argument("--enable-trail", action="store_true")
    ap.add_argument("--trail-type", default="stc_keep_trail")
    ap.add_argument("--trail-threshold-us", type=int, default=10000)
    ap.add_argument("--stderr-log", default="")
    args = ap.parse_args()

    it = HalEventPipeIterator(
        bridge_bin=args.bridge_bin,
        serial=args.serial,
        slice_us=args.slice_us,
        wait_mode=args.wait_mode,
        poll_sleep_us=args.poll_sleep_us,
        enable_trigger_in=args.enable_trigger_in,
        enable_erc=args.enable_erc,
        erc_rate=args.erc_rate,
        enable_trail=args.enable_trail,
        trail_type=args.trail_type,
        trail_threshold_us=args.trail_threshold_us,
        stderr_log=args.stderr_log,
    )
    try:
        height, width = it.get_size()
        started = time.time()
        total_events = 0
        total_triggers = 0
        rows = []
        for idx, evs in zip(range(max(1, args.slices)), it):
            trigs = it.get_ext_trigger_events()
            row = {
                "idx": idx,
                "current_time_us": it.get_current_time(),
                "events": int(len(evs)),
                "triggers": int(len(trigs)),
                "event_dtype": str(evs.dtype),
                "trigger_dtype": str(getattr(trigs, "dtype", "")),
            }
            rows.append(row)
            total_events += int(len(evs))
            total_triggers += int(len(trigs))
            it.clear_ext_trigger_events()
        elapsed = max(1e-6, time.time() - started)
        print(json.dumps({
            "ok": True,
            "serial": args.serial,
            "size": [height, width],
            "slices": len(rows),
            "total_events": total_events,
            "total_triggers": total_triggers,
            "events_per_s": total_events / elapsed,
            "rows": rows[:5],
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        it.close()


if __name__ == "__main__":
    raise SystemExit(main())
