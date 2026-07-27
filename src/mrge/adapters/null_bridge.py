# SPDX-License-Identifier: AGPL-3.0-only
"""A no-op bridge that makes network absence explicit."""

from collections.abc import Mapping


class NullBridge:
    def __init__(self) -> None:
        self.messages: list[Mapping[str, object]] = []

    def publish(self, message: Mapping[str, object]) -> None:
        self.messages.append(dict(message))
