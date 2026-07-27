# SPDX-License-Identifier: AGPL-3.0-only
"""Replaceable research boundaries for perception, sources, and bridges."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from mrge.contracts import Frame


@runtime_checkable
class InferenceBackend(Protocol):
    def predict(self, frame: Frame) -> Sequence[Mapping[str, object]]: ...


@runtime_checkable
class VisibleCameraAdapter(Protocol):
    def frames(self) -> Iterable[Frame]: ...


@runtime_checkable
class EventAdapter(Protocol):
    def frames(self) -> Iterable[Frame]: ...


@runtime_checkable
class GameBridge(Protocol):
    def publish(self, message: Mapping[str, object]) -> None: ...
