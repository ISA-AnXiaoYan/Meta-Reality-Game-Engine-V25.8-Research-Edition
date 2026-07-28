# SPDX-License-Identifier: AGPL-3.0-only
"""Verify the V25.8/V26 historical import and write its public manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path


EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".pyd", ".dll", ".so", ".lib", ".exe",
    ".pt", ".pth", ".onnx", ".engine",
}
V25_INCLUDED_TREES = ("legacy", "datasets", "evidence", "releases", "runtime")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path, relative_roots: Iterable[str] | None = None) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    roots = [root / item for item in relative_roots] if relative_roots else [root]
    for selected in roots:
        if not selected.exists():
            continue
        for path in selected.rglob("*"):
            if not path.is_file() or any(part in {".git", ".ua", ".understand-anything", "__pycache__"} for part in path.parts):
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            result[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def git_files(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True)
    return [line for line in output.splitlines() if line]


def verify(source: dict[str, dict[str, object]], target: dict[str, dict[str, object]], label: str) -> dict[str, object]:
    missing = sorted(set(source) - set(target))
    unexpected = sorted(set(target) - set(source))
    changed = sorted(key for key in set(source) & set(target) if source[key] != target[key])
    if missing or unexpected or changed:
        raise SystemExit(
            f"{label} import mismatch: missing={len(missing)} unexpected={len(unexpected)} changed={len(changed)}"
        )
    return {
        "files": len(source),
        "bytes": sum(int(item["bytes"]) for item in source.values()),
        "tree_sha256": hashlib.sha256(
            "".join(f"{name}\0{item['sha256']}\n" for name, item in sorted(source.items())).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v25-source", type=Path, required=True)
    parser.add_argument("--v25-target", type=Path, required=True)
    parser.add_argument("--v26-source", type=Path, required=True)
    parser.add_argument("--v26-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v25_source = files_under(args.v25_source, V25_INCLUDED_TREES)
    v25_target = files_under(args.v25_target)
    v26_source_all = files_under(args.v26_source)
    v26_names = set(git_files(args.v26_source))
    v26_source = {name: item for name, item in v26_source_all.items() if name in v26_names}
    v26_target = files_under(args.v26_target)

    manifest = {
        "schema_version": "mrge.historical-release-manifest.v1",
        "release_mode": "full_public_release_with_authorized_exceptions",
        "v25_8": {
            "source_kind": "frozen filesystem snapshot",
            "included_trees": list(V25_INCLUDED_TREES),
            "excluded_trees": ["components", "workspace", "worktrees"],
            **verify(v25_source, v25_target, "V25.8"),
        },
        "v26": {
            "source_kind": "Git-tracked checkout",
            "source_commit": subprocess.check_output(
                ["git", "-C", str(args.v26_source), "rev-parse", "HEAD"], text=True
            ).strip(),
            **verify(v26_source, v26_target, "V26"),
        },
        "non_redistributable_exceptions": [
            "vendor SDKs and binaries",
            "third-party model weights",
            "credentials, certificates and tokens",
            "material without confirmed redistribution authority",
        ],
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
