# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic synthetic frame source used by tests and demos."""

from mrge.contracts import Frame


def frames(count: int = 3):
    for index in range(count):
        yield Frame("synthetic-run", "synthetic-epoch", "cam-01", index, index * 1_000_000, {"objects": []})
