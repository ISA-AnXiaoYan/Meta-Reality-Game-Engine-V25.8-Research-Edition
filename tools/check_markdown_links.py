# SPDX-License-Identifier: AGPL-3.0-only
"""Validate repository-local Markdown links without network access."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#")


def tracked_markdown() -> list[Path]:
    names = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).splitlines()
    return [ROOT / name for name in names]


def main() -> int:
    failures: list[str] = []
    for path in tracked_markdown():
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().strip("<>")
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            target = target.split("#", 1)[0].strip()
            if not target:
                continue
            destination = (path.parent / target).resolve()
            try:
                destination.relative_to(ROOT)
            except ValueError:
                failures.append(f"{path.relative_to(ROOT)}: link escapes repository: {target}")
                continue
            if not destination.exists():
                failures.append(f"{path.relative_to(ROOT)}: missing target: {target}")
    if failures:
        print("\n".join(failures))
        return 1
    print("markdown links: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
