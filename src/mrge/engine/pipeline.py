# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal research pipeline; candidate output is never final authority."""

from mrge.contracts import Frame, HitCandidate


def process(frame: Frame) -> HitCandidate:
    return HitCandidate(frame.run_id, frame.epoch_id, "unknown", "synthetic", 0.0)
