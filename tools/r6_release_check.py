# SPDX-License-Identifier: AGPL-3.0-only
"""Dependency-free R6 release checks for the research edition."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = {".py", ".toml", ".yml", ".yaml", ".json"}
SECRET_PATTERNS = [re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY-----")]


def tracked() -> list[Path]:
    names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    return [ROOT / name for name in names]


def main() -> int:
    files = tracked()
    missing_spdx = []
    absolute_paths = []
    secret_hits = []
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in CODE_SUFFIXES and path.name not in {"pyproject.toml", "RESEARCH_RELEASE_MANIFEST.json", "SOURCE_PROVENANCE.json", "RENAME_APPROVAL.json"}:
            if "SPDX-License-Identifier" not in text:
                missing_spdx.append(str(path.relative_to(ROOT)))
        if re.search(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/)", text):
            absolute_paths.append(str(path.relative_to(ROOT)))
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                secret_hits.append(str(path.relative_to(ROOT)))
    result = {
        "tracked_files": len(files),
        "root_pyproject_count": sum(path.name == "pyproject.toml" and path.parent == ROOT for path in files),
        "missing_spdx": missing_spdx,
        "absolute_path_hits": absolute_paths,
        "secret_hits": secret_hits,
        "real_data_or_weights": [str(p.relative_to(ROOT)) for p in files if re.search(r"(?i)(weights|datasets?)/|\\.(pt|onnx|engine)$", str(p))],
        "status": "PASS" if not (missing_spdx or absolute_paths or secret_hits) else "FAIL",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
