# SPDX-License-Identifier: AGPL-3.0-only
"""Build a hash-only pre-V27 source inventory without copying source content."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

EXCLUDED_PARTS = {"archive", "archives", "data", "datasets", "weights", "vendor", "sdk", "sync_ipc", "runtime", "secrets"}
CODE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".md", ".json", ".yaml", ".yml", ".toml", ".ps1", ".sh"}


def classify(source_id: str, relative: str, suffix: str) -> tuple[str, str]:
    parts = {part.lower() for part in Path(relative).parts}
    lowered = relative.lower()
    if any(token in lowered for token in ("proprietary", "private", "secret", "credential", "model_weight")):
        return "BLOCKED", "LICENSE_OR_SECRET_BOUND"
    if parts & EXCLUDED_PARTS:
        return "BLOCKED", "PRIVATE_OR_VENDOR_BOUND"
    if source_id == "v27_clean_runtime_reference":
        return "REFERENCE_ONLY", "V27_REFERENCE_UNLESS_ORIGINALITY_APPROVED"
    if suffix.lower() not in CODE_SUFFIXES:
        return "REVIEW_REQUIRED", "NON_CODE_OR_ASSET_REVIEW"
    return "REVIEW_REQUIRED", "ORIGINALITY_AND_LICENSE_REVIEW"


def inventory(source_id: str, root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if {part.lower() for part in Path(relative).parts} & EXCLUDED_PARTS:
            continue
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        status, reason = classify(source_id, relative, path.suffix)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        yield {
            "source_id": source_id,
            "relative_path": relative,
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "status": status,
            "reason": reason,
            "target": "pending-review",
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="source_id=path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = []
    for spec in args.source:
        source_id, raw_root = spec.split("=", 1)
        rows.extend(inventory(source_id, Path(raw_root)))
    rows.sort(key=lambda row: (row["source_id"], row["relative_path"]))
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["source_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"inventory_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
