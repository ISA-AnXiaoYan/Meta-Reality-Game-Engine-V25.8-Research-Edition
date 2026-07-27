# SPDX-License-Identifier: Apache-2.0
"""Small JSONL replay reader/writer with no runtime dependencies."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def write(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def read(path: str | Path) -> Iterator[dict[str, Any]]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)
