# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal research pipeline; candidate output is never final authority."""

from collections.abc import Iterable

from mrge.contracts import Frame, HitCandidate, ResultEnvelope


def process(frame: Frame) -> HitCandidate:
    return HitCandidate(frame.run_id, frame.epoch_id, "unknown", "synthetic", 0.0)


def run(frames: Iterable[Frame]) -> list[ResultEnvelope]:
    """Process every input fact and emit one explicit terminal result."""
    return [
        ResultEnvelope(
            run_id=frame.run_id,
            epoch_id=frame.epoch_id,
            terminal_state="candidate_emitted",
            candidate=process(frame),
        )
        for frame in frames
    ]
