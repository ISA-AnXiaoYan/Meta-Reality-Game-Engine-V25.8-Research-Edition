# SPDX-License-Identifier: AGPL-3.0-only
"""Research-only command line entry point."""

import argparse
import json
from pathlib import Path
from mrge.adapters.synthetic import frames
from mrge.engine.pipeline import process, run
from mrge.engine.reports import build_reports
from replay.jsonl import write
from replay.jsonl import read
from replay.faults import inject


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mrge")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sim = sub.add_parser("simulate")
    sim.add_argument("--frames", type=int, default=3)
    sim.add_argument("--cameras", type=int, default=1)
    sample = sub.add_parser("generate-sample")
    sample.add_argument("--output", required=True)
    sample.add_argument("--cameras", type=int, default=1)
    sample.add_argument("--frames", type=int, default=3)
    report = sub.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--input", required=True)
    rep.add_argument("--fault", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps({"status": "ok", "authority": "research-only"}))
    elif args.command == "simulate":
        for envelope in run(frames(args.frames, args.cameras)):
            print(envelope.to_json())
    elif args.command == "generate-sample":
        write(args.output, ({"frame": json.loads(frame.to_json())} for frame in frames(args.frames, args.cameras)))
        print(json.dumps({"written": str(Path(args.output)), "official_result": False}, sort_keys=True))
    elif args.command == "report":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for name, payload in build_reports(read(args.input)).items():
            (output / f"{name}.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "reports": 5, "official_result": False}, sort_keys=True))
    else:
        rows = inject(read(args.input), args.fault)
        for row in rows:
            print(json.dumps({"replayed": row, "terminal_state": "replayed", "official_result": False}, sort_keys=True))
    return 0
