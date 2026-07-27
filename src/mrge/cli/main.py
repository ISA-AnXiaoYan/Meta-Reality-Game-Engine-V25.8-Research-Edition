# SPDX-License-Identifier: AGPL-3.0-only
"""Research-only command line entry point."""

import argparse
import json
from pathlib import Path
from mrge.adapters.synthetic import frames
from mrge.engine.pipeline import process, run
from replay.jsonl import write
from replay.jsonl import read


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mrge")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sim = sub.add_parser("simulate")
    sim.add_argument("--frames", type=int, default=3)
    sample = sub.add_parser("generate-sample")
    sample.add_argument("--output", required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps({"status": "ok", "authority": "research-only"}))
    elif args.command == "simulate":
        for envelope in run(frames(args.frames)):
            print(envelope.to_json())
    elif args.command == "generate-sample":
        write(args.output, ({"frame": json.loads(frame.to_json())} for frame in frames(3)))
        print(json.dumps({"written": str(Path(args.output)), "official_result": False}, sort_keys=True))
    else:
        for row in read(args.input):
            print(json.dumps({"replayed": row, "terminal_state": "replayed", "official_result": False}, sort_keys=True))
    return 0
