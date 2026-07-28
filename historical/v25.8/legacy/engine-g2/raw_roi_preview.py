from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from marker_blink_decoder import iter_raw


def _select_polarity(evs: np.ndarray, polarity: str) -> np.ndarray:
    if polarity in ("positive", "pos"):
        return evs[evs["p"] > 0]
    if polarity in ("negative", "neg"):
        return evs[evs["p"] <= 0]
    return evs


def _window_scores(heat: np.ndarray, *, window: int, stride: int, top_n: int) -> list[dict]:
    height, width = heat.shape
    integral = np.pad(heat.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    candidates: list[dict] = []
    for y in range(0, max(1, height - window + 1), stride):
        for x in range(0, max(1, width - window + 1), stride):
            y2 = min(height, y + window)
            x2 = min(width, x + window)
            score = int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])
            candidates.append({"x": int(x), "y": int(y), "w": int(x2 - x), "h": int(y2 - y), "score": score})
    return sorted(candidates, key=lambda row: int(row["score"]), reverse=True)[: max(1, int(top_n))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest bounded LED ROIs from a Metavision RAW recording.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--polarity", choices=["any", "positive", "negative", "pos", "neg"], default="any")
    parser.add_argument("--raw-delta-t-us", type=int, default=4000)
    parser.add_argument("--max-events-per-slice", type=int, default=4000)
    parser.add_argument("--limit-slices", type=int, default=0)
    args = parser.parse_args(argv)

    width = max(1, int(args.width))
    height = max(1, int(args.height))
    heat = np.zeros((height, width), dtype=np.int64)
    slices = 0
    events_total = 0
    events_used = 0
    for evs in iter_raw(str(args.raw), delta_t_us=int(args.raw_delta_t_us)):
        slices += 1
        events_total += int(len(evs))
        selected = _select_polarity(np.asarray(evs), str(args.polarity))
        limit = max(0, int(args.max_events_per_slice))
        if limit and len(selected) > limit:
            step = max(1, int(np.ceil(len(selected) / float(limit))))
            selected = selected[::step]
        if len(selected):
            xs = selected["x"].astype(np.int64)
            ys = selected["y"].astype(np.int64)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            np.add.at(heat, (ys[valid], xs[valid]), 1)
            events_used += int(np.count_nonzero(valid))
        if int(args.limit_slices) and slices >= int(args.limit_slices):
            break

    peak_y, peak_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    report = {
        "schema_version": 1,
        "kind": "ir_id_raw_roi_preview",
        "raw": str(args.raw),
        "width": width,
        "height": height,
        "polarity": str(args.polarity),
        "slices": slices,
        "events_total": events_total,
        "events_used": events_used,
        "peak": {"x": int(peak_x), "y": int(peak_y), "count": int(heat[peak_y, peak_x])},
        "suggested_rois": _window_scores(
            heat,
            window=max(1, int(args.window)),
            stride=max(1, int(args.stride)),
            top_n=max(1, int(args.top_n)),
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
