# SPDX-License-Identifier: Apache-2.0
"""Stable, dependency-free research contracts."""

from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass(frozen=True)
class Frame:
    run_id: str
    epoch_id: str
    camera_id: str
    frame_index: int
    timestamp_ns: int
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class HitCandidate:
    run_id: str
    epoch_id: str
    person_id: str
    source: str
    confidence: float
    authority: str = "candidate"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ResultEnvelope:
    """A terminal research result; never a production verdict."""

    run_id: str
    epoch_id: str
    terminal_state: str
    candidate: HitCandidate
    official_result: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
