# SPDX-License-Identifier: AGPL-3.0-only
"""Build a deterministic, network-free JSONL research sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrge.adapters.synthetic import frames
from replay.jsonl import write


def build(count: int, cameras: int = 1) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "mrge.synthetic_sample.v1",
            "frame": json.loads(frame.to_json()),
            "official_result": False,
        }
        for frame in frames(count, cameras)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--cameras", type=int, default=1)
    args = parser.parse_args(argv)
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    write(Path(args.output), build(args.frames, args.cameras))
    print(json.dumps({"output": str(args.output), "frames": args.frames, "cameras": args.cameras, "official_result": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
