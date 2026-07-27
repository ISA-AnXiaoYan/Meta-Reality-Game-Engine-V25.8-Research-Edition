# SPDX-License-Identifier: AGPL-3.0-only
"""Research-only command line entry point."""

import argparse
import json
from mrge.adapters.synthetic import frames
from mrge.engine.pipeline import process
from replay.jsonl import read


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mrge")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sim = sub.add_parser("simulate")
    sim.add_argument("--frames", type=int, default=3)
    rep = sub.add_parser("replay")
    rep.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps({"status": "ok", "authority": "research-only"}))
    elif args.command == "simulate":
        for frame in frames(args.frames):
            print(process(frame).to_json())
    else:
        for row in read(args.input):
            print(json.dumps({"replayed": row}, sort_keys=True))
    return 0
