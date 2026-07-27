# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json

from tools.build_synthetic_sample import build


def test_synthetic_sample_is_deterministic():
    first = json.dumps(build(3), sort_keys=True, separators=(",", ":"))
    second = json.dumps(build(3), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()
    assert all(row["official_result"] is False for row in build(3))
