# SPDX-License-Identifier: AGPL-3.0-only
"""Null adapter used to make the absence of hardware explicit."""


class NullAdapter:
    """An inert source; it never opens devices or reads profiles."""

    def frames(self):
        return iter(())
