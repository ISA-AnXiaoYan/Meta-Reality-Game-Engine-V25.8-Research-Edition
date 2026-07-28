from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _load_json_lines(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_perf(path: Optional[Path]) -> List[Dict[str, object]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _float_value(row: Dict[str, object], key: str) -> Optional[float]:
    try:
        value = row.get(key)
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def validate(
    *,
    observation_path: Path,
    status_path: Path,
    perf_path: Optional[Path],
    expected_id: int,
    run_id: str,
    min_observations: int,
    max_false_ids: int,
    max_queue_dropped: int,
    max_p95_ms: float,
) -> Dict[str, object]:
    rows = _load_json_lines(observation_path)
    status = _load_json(status_path)
    perf_rows = _load_perf(perf_path)
    marker_ids = [int(row.get("marker_id", -1)) for row in rows]
    false_ids = [marker_id for marker_id in marker_ids if marker_id != int(expected_id)]
    run_id_mismatches = sum(str(row.get("run_id", "")) != str(run_id) for row in rows)
    stale_rows = sum(bool(row.get("stale", False)) for row in rows)
    schema_errors = sum(int(row.get("schema_version", -1)) != 2 for row in rows)
    protocol_errors = sum(str(row.get("protocol_version", "")) != "ir-id-blink-v1" for row in rows)
    hash_errors = sum(not str(row.get("decoder_config_hash", "")).startswith("sha256:") for row in rows)
    queue_dropped = int(status.get("queue_dropped", 0) or 0)
    status_enabled = bool(status.get("enabled", False))
    status_error = str(status.get("last_error", "") or "")
    p95_values = [value for row in perf_rows if (value := _float_value(row, "process_ms_p95")) is not None]
    max_observed_p95 = max(p95_values) if p95_values else None
    checks = {
        "observation_file_present": observation_path.exists(),
        "status_file_present": status_path.exists(),
        "status_enabled": status_enabled,
        "status_error_empty": not status_error,
        "expected_observation_count": len(marker_ids) >= max(1, int(min_observations)),
        "false_id_limit": len(false_ids) <= max(0, int(max_false_ids)),
        "run_id_match": run_id_mismatches == 0,
        "schema_ok": schema_errors == 0,
        "protocol_ok": protocol_errors == 0,
        "config_hash_present": hash_errors == 0,
        "no_stale_observation": stale_rows == 0,
        "queue_drop_limit": queue_dropped <= max(0, int(max_queue_dropped)),
        "perf_present": bool(p95_values),
        "perf_p95_limit": bool(max_observed_p95 is not None and max_observed_p95 <= float(max_p95_ms)),
    }
    return {
        "schema_version": 1,
        "kind": "ir_id_minimum_field_validation",
        "passed": all(checks.values()),
        "checks": checks,
        "inputs": {
            "observation": str(observation_path),
            "status": str(status_path),
            "perf": "" if perf_path is None else str(perf_path),
        },
        "expected": {
            "marker_id": int(expected_id),
            "run_id": str(run_id),
            "min_observations": int(min_observations),
            "max_false_ids": int(max_false_ids),
            "max_queue_dropped": int(max_queue_dropped),
            "max_p95_ms": float(max_p95_ms),
        },
        "observed": {
            "observation_count": len(marker_ids),
            "marker_ids": marker_ids,
            "false_ids": false_ids,
            "run_id_mismatch_count": int(run_id_mismatches),
            "stale_row_count": int(stale_rows),
            "schema_error_count": int(schema_errors),
            "protocol_error_count": int(protocol_errors),
            "config_hash_error_count": int(hash_errors),
            "queue_dropped": int(queue_dropped),
            "status_state": str(status.get("state", "")),
            "status_error": status_error,
            "perf_samples": len(p95_values),
            "max_process_p95_ms": max_observed_p95,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a V26-D minimum IR-ID field run.")
    parser.add_argument("--observation", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--perf", required=True)
    parser.add_argument("--expected-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-false-ids", type=int, default=0)
    parser.add_argument("--max-queue-dropped", type=int, default=0)
    parser.add_argument("--max-p95-ms", type=float, default=2.0)
    parser.add_argument("--output", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate(
        observation_path=Path(args.observation),
        status_path=Path(args.status),
        perf_path=Path(args.perf),
        expected_id=int(args.expected_id),
        run_id=str(args.run_id),
        min_observations=int(args.min_observations),
        max_false_ids=int(args.max_false_ids),
        max_queue_dropped=int(args.max_queue_dropped),
        max_p95_ms=float(args.max_p95_ms),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
