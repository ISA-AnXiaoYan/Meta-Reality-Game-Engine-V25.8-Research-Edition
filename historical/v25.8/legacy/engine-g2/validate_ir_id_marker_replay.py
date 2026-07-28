from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_ids(value: str) -> List[int]:
    ids = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not ids:
        raise ValueError("--expected-ids must not be empty")
    return ids


def validate(input_path: Path, expected_ids: List[int]) -> dict:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed_ids = [int(row["marker_id"]) for row in rows]
    schema_errors = sum(int(row.get("schema_version", -1)) != 2 for row in rows)
    protocol_errors = sum(row.get("protocol_version") != "ir-id-blink-v1" for row in rows)
    missing_config_hash = sum(not str(row.get("decoder_config_hash", "")).startswith("sha256:") for row in rows)
    stale_rows = sum(bool(row.get("stale", False)) for row in rows)
    sequence_match = observed_ids == list(expected_ids)
    false_ids = sum(observed not in set(expected_ids) for observed in observed_ids)
    passed = bool(
        sequence_match
        and schema_errors == 0
        and protocol_errors == 0
        and missing_config_hash == 0
        and stale_rows == 0
    )
    return {
        "schema_version": 1,
        "kind": "ir_id_marker_replay_validation",
        "input": input_path.name,
        "expected_ids": list(expected_ids),
        "observed_ids": observed_ids,
        "observation_count": len(rows),
        "sequence_match": sequence_match,
        "false_id_count": int(false_ids),
        "schema_error_count": int(schema_errors),
        "protocol_error_count": int(protocol_errors),
        "missing_config_hash_count": int(missing_config_hash),
        "stale_row_count": int(stale_rows),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a deterministic IR-ID marker replay.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-ids", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = validate(Path(args.input), parse_ids(args.expected_ids))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
