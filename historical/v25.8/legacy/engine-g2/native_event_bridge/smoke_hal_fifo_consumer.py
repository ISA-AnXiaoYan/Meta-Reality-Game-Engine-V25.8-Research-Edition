#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from hal_event_pipe_iterator import HalEventFifoIterator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fifo", required=True)
    ap.add_argument("--slices", type=int, default=200)
    ap.add_argument("--alias", default="")
    args = ap.parse_args()

    it = HalEventFifoIterator(fifo_path=args.fifo)
    try:
        height, width = it.get_size()
        started = time.time()
        total_events = 0
        total_triggers = 0
        first_rows = []
        for idx, evs in zip(range(max(1, args.slices)), it):
            trigs = it.get_ext_trigger_events()
            row = {
                "idx": idx,
                "current_time_us": it.get_current_time(),
                "events": int(len(evs)),
                "triggers": int(len(trigs)),
            }
            if len(first_rows) < 5:
                first_rows.append(row)
            total_events += int(len(evs))
            total_triggers += int(len(trigs))
            it.clear_ext_trigger_events()
        elapsed = max(1e-6, time.time() - started)
        print(json.dumps({
            "ok": True,
            "alias": args.alias,
            "fifo": args.fifo,
            "size": [height, width],
            "slices": max(1, args.slices),
            "total_events": total_events,
            "total_triggers": total_triggers,
            "events_per_s": total_events / elapsed,
            "first_rows": first_rows,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        it.close()


if __name__ == "__main__":
    raise SystemExit(main())
