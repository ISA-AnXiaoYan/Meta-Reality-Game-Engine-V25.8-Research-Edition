# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_v26_public_schemas_keep_authority_boundaries():
    recording = json.loads((ROOT / "contracts/recording_bundle.schema.json").read_text(encoding="utf-8"))
    polygon = json.loads((ROOT / "contracts/polygon_gate.schema.json").read_text(encoding="utf-8"))
    ir_id = json.loads((ROOT / "contracts/ir_id.schema.json").read_text(encoding="utf-8"))
    assert recording["properties"]["authority"]["const"] == "research-only"
    assert polygon["properties"]["authority"]["const"] == "candidate"
    assert ir_id["properties"]["authority"]["const"] == "shadow"
